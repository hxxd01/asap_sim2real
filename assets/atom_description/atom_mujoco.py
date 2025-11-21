import mujoco, mujoco_viewer
import numpy as np
import onnxruntime
import yaml
import os
import joblib
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import torch
import pickle
from datetime import datetime
import os


def read_conf(config_file):
    cfg = SimpleNamespace()
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    cfg.policy_path = config["policy_path"]
    cfg.cycle_time = config["cycle_time"]
    cfg.xml_path = config["xml_path"]
    cfg.num_single_obs = config["num_single_obs"]
    cfg.simulation_dt = config["simulation_dt"]
    cfg.simulation_duration = config["simulation_duration"]
    cfg.control_decimation = config["control_decimation"]
    cfg.frame_stack = config["frame_stack"]
    cfg.episode_steps = config["episode_steps"]
    cfg.total_steps = config["total_steps"]
    cfg.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)
    cfg.obs_scale_base_ang_vel = config["obs_scale_base_ang_vel"]
    cfg.obs_scale_dof_pos = config["obs_scale_dof_pos"]
    cfg.obs_scale_dof_vel = config["obs_scale_dof_vel"]
    cfg.obs_scale_gvec = config["obs_scale_gvec"]
    cfg.obs_scale_refmotion = config["obs_scale_refmotion"]
    cfg.obs_scale_hist = config["obs_scale_hist"]
    cfg.num_actions = config["num_actions"]
    cfg.clip_observations = config["clip_observations"]
    cfg.clip_actions = config["clip_actions"]

    # 支持 per-joint action_scale（可以是单一值或数组）
    action_scale_raw = config["action_scale"]
    if isinstance(action_scale_raw, (list, tuple)):
        cfg.action_scale = np.array(action_scale_raw, dtype=np.float32)
        print(f"✅ 使用 per-joint action_scale: min={cfg.action_scale.min():.4f}, max={cfg.action_scale.max():.4f}")
    else:
        cfg.action_scale = np.full(cfg.num_actions, action_scale_raw, dtype=np.float32)
        print(f"✅ 使用统一 action_scale: {action_scale_raw}")

    cfg.kps = np.array(config["kps"], dtype=np.float32)
    cfg.kds = np.array(config["kds"], dtype=np.float32)
    cfg.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
    cfg.render = config["render"]
    cfg.use_noise = config["use_noise"]
    cfg.log_torque = config.get("log_torque", False)
    cfg.init_pos = np.array(config["init_pos"], dtype=np.float32)

    # 添加termination相关配置
    cfg.use_termination = config.get("use_termination", True)
    cfg.termination_gravity_x = config.get("termination_gravity_x", 0.85)
    cfg.termination_gravity_y = config.get("termination_gravity_y", 0.85)
    # 新增：最小高度与力矩范数阈值
    cfg.termination_min_height = config.get("termination_min_height", 0.35)

    # 添加数据处理配置
    cfg.min_episode_length = config.get("min_episode_length", 75)
    cfg.auto_process = config.get("auto_process", True)

    # 可选：初始化基座位姿（MuJoCo qpos: [base_xyz(3), base_quat_wxyz(4), joints...]）
    cfg.init_base_pos = np.array(config.get("init_base_pos", [0, 0, 0.95]), dtype=np.float32)
    # MuJoCo quaternion format: [w, x, y, z] - identity quaternion is [1, 0, 0, 0]
    cfg.init_base_quat_wxyz = np.array(config.get("init_base_quat_wxyz", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)

    return cfg


def get_mujoco_data(data):
    mujoco_data = {}
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    # MuJoCo qpos[3:7] = [w, x, y, z]
    # scipy R.from_quat expects [x, y, z, w]
    quat_xyzw = np.array([q[4], q[5], q[6], q[3]])  # convert to x,y,z,w
    r = R.from_quat(quat_xyzw)
    base_angvel = dq[3:6]
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    mujoco_data['mujoco_dof_pos'] = q[7:]
    mujoco_data['mujoco_dof_vel'] = dq[6:]
    mujoco_data['mujoco_base_angvel'] = base_angvel
    mujoco_data['mujoco_gvec'] = gvec
    mujoco_data['base_height'] = q[2]
    mujoco_data['base_quat_wxyz'] = np.array([q[3], q[4], q[5], q[6]])  # w,x,y,z for debug (MuJoCo format)
    return mujoco_data


def get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg):
    """
    构建 ATOM 机器人观测（27 DOF）
    单帧观测：actions(27) + base_ang_vel(3) + dof_pos(27) + dof_vel(27) + gravity(3) + phase(1) = 88
    完整观测：单帧 + 历史(352) = 440
    """
    mujoco_base_angvel = mujoco_data["mujoco_base_angvel"]
    mujoco_dof_pos = mujoco_data["mujoco_dof_pos"]
    mujoco_dof_vel = mujoco_data["mujoco_dof_vel"]
    mujoco_gvec = mujoco_data["mujoco_gvec"]

    # ATOM 27 DOF 噪声
    if cfg.use_noise:
        noise_base_ang_vel = (np.random.rand(3) * 2. - 1.) * 0.3
        noise_projected_gravity = (np.random.rand(3) * 2. - 1.) * 0.2
        noise_dof_pos = (np.random.rand(27) * 2. - 1.) * 0.01
        noise_dof_vel = (np.random.rand(27) * 2. - 1.) * 1.0
    else:
        noise_base_ang_vel = np.zeros(3)
        noise_projected_gravity = np.zeros(3)
        noise_dof_pos = np.zeros(27)
        noise_dof_vel = np.zeros(27)

    # 相位在0到1之间循环
    ref_motion_phase = ((counter + 1) * cfg.simulation_dt / cfg.cycle_time) % 1.0
    
    # 🔍 Debug: 打印前5步的详细观测数据
    control_step = counter // cfg.control_decimation
    if control_step < 5:
        quat_wxyz = mujoco_data.get('base_quat_wxyz', [0,0,0,1])
        action_arr = np.asarray(action).reshape(-1)  # 确保是1D数组
        print(f"\n[Python Step {control_step} Observation Debug]")
        print(f"  quaternion (w,x,y,z): {quat_wxyz[0]:.6f}, {quat_wxyz[1]:.6f}, {quat_wxyz[2]:.6f}, {quat_wxyz[3]:.6f}")
        print(f"  ang_vel: {float(mujoco_base_angvel[0]):.6f}, {float(mujoco_base_angvel[1]):.6f}, {float(mujoco_base_angvel[2]):.6f}")
        print(f"  dof_pos[0-2]: {float(mujoco_dof_pos[0]):.6f}, {float(mujoco_dof_pos[1]):.6f}, {float(mujoco_dof_pos[2]):.6f}")
        print(f"  dof_vel[0-2]: {float(mujoco_dof_vel[0]):.6f}, {float(mujoco_dof_vel[1]):.6f}, {float(mujoco_dof_vel[2]):.6f}")
        print(f"  last_action[0-2]: {float(action_arr[0]):.6f}, {float(action_arr[1]):.6f}, {float(action_arr[2]):.6f}")
        print(f"  projected_gravity: {float(mujoco_gvec[0]):.6f}, {float(mujoco_gvec[1]):.6f}, {float(mujoco_gvec[2]):.6f}")
        # 计算 deltaJointPos
        delta_pos = mujoco_dof_pos - cfg.default_dof_pos
        print(f"  deltaJointPos[0-2]: {float(delta_pos[0]):.6f}, {float(delta_pos[1]):.6f}, {float(delta_pos[2]):.6f}")
        print(f"  motion_phase: {ref_motion_phase:.6f}")


    # 单帧观测维度：88
    obs_sigle = np.zeros([1, cfg.num_single_obs], dtype=np.float32)
    # actions: 0:27
    obs_sigle[0, 0:27] = action
    # base_ang_vel: 27:30
    obs_sigle[0, 27:30] = (mujoco_base_angvel + noise_base_ang_vel) * cfg.obs_scale_base_ang_vel
    # dof_pos: 30:57
    dof_pos = mujoco_dof_pos - cfg.default_dof_pos
    obs_sigle[0, 30:57] = (dof_pos + noise_dof_pos) * cfg.obs_scale_dof_pos
    # dof_vel: 57:84
    obs_sigle[0, 57:84] = (mujoco_dof_vel + noise_dof_vel) * cfg.obs_scale_dof_vel
    # projected_gravity: 84:87
    obs_sigle[0, 84:87] = (mujoco_gvec + noise_projected_gravity) * cfg.obs_scale_gvec
    # ref_motion_phase: 87:88
    obs_sigle[0, 87] = ref_motion_phase * cfg.obs_scale_refmotion

    # 完整观测维度：440 = 27(actions) + 3(ang_vel) + 27(dof_pos) + 27(dof_vel) + 352(hist) + 3(gravity) + 1(phase)
    num_obs_full = 27 + 3 + 27 + 27 + 352 + 3 + 1  # 440
    obs_all = np.zeros([1, num_obs_full], dtype=np.float32)

    # 组装完整观测（按训练时的顺序）
    idx = 0
    # actions: 0:27
    obs_all[0, idx:idx + 27] = obs_sigle[0, 0:27].copy()
    idx += 27
    # base_ang_vel: 27:30
    obs_all[0, idx:idx + 3] = obs_sigle[0, 27:30].copy()
    idx += 3
    # dof_pos: 30:57
    obs_all[0, idx:idx + 27] = obs_sigle[0, 30:57].copy()
    idx += 27
    # dof_vel: 57:84
    obs_all[0, idx:idx + 27] = obs_sigle[0, 57:84].copy()
    idx += 27
    # history_actor: 84:436 (352 维)
    obs_all[0, idx:idx + 352] = hist_obs_c[0] * cfg.obs_scale_hist
    idx += 352
    # projected_gravity: 436:439
    obs_all[0, idx:idx + 3] = obs_sigle[0, 84:87].copy()
    idx += 3
    # ref_motion_phase: 439:440
    obs_all[0, idx] = obs_sigle[0, 87].copy()

    hist_obs_cat = update_hist_obs(hist_dict, obs_sigle)
    obs_all = np.clip(obs_all, -cfg.clip_observations, cfg.clip_observations)

    return obs_all, hist_obs_cat


def update_hist_obs(hist_dict, obs_sigle):
    """
    更新历史观测（ATOM 27 DOF）
    历史维度：27*4 + 3*4 + 27*4 + 27*4 + 3*4 + 1*4 = 352
    """
    slices = {
        'actions': slice(0, 27),
        'base_ang_vel': slice(27, 30),
        'dof_pos': slice(30, 57),
        'dof_vel': slice(57, 84),
        'projected_gravity': slice(84, 87),
        'ref_motion_phase': slice(87, 88)
    }

    for key, slc in slices.items():
        arr = np.delete(hist_dict[key], -1, axis=0)
        arr = np.vstack((obs_sigle[0, slc], arr))
        hist_dict[key] = arr

    hist_obs = np.concatenate([
        hist_dict[key].reshape(1, -1)
        for key in sorted(hist_dict.keys())
    ], axis=1).astype(np.float32)
    return hist_obs


def pd_control(target_pos, dof_pos, target_vel, dof_vel, cfg):
    torque_out = (target_pos - dof_pos) * cfg.kps + (target_vel - dof_vel) * cfg.kds
    return torque_out


def parse_dof_axis_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    joints = root.findall('.//joint')
    dof_axis = []
    for j in joints:
        if 'type' in j.attrib and j.attrib['type'] in ['free', 'float']:
            continue
        axis_str = j.attrib.get('axis', None)
        if axis_str is not None:
            axis = [float(x) for x in axis_str.strip().split()]
            dof_axis.append(axis)
    return np.array(dof_axis, dtype=np.float32)


def parse_joint_names_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    joints = root.findall('.//joint')
    names = []
    for j in joints:
        jtype = j.attrib.get('type', '')
        if jtype in ['free', 'float']:
            continue
        name = j.attrib.get('name', None)
        if name is not None:
            names.append(name)
    return names


def check_termination(mujoco_data, cfg, counter):
    if not cfg.use_termination:
        return False, None

    mujoco_gvec = mujoco_data["mujoco_gvec"]
    gravity_x_violation = abs(mujoco_gvec[0]) > cfg.termination_gravity_x
    gravity_y_violation = abs(mujoco_gvec[1]) > cfg.termination_gravity_y
    gravity_termination = gravity_x_violation or gravity_y_violation
    if gravity_termination:
        print(f"[Termination] Step {counter}: gravity_x={mujoco_gvec[0]:.3f}(>{cfg.termination_gravity_x}), "
              f"gravity_y={mujoco_gvec[1]:.3f}(>{cfg.termination_gravity_y})")
        return True, "gravity"

    # 高度终止：底座高度过低
    base_h = mujoco_data.get("base_height", None)
    if base_h is not None and base_h < cfg.termination_min_height:
        print(f"[Termination] Step {counter}: base_height={base_h:.3f}(<{cfg.termination_min_height})")
        return True, "height"

    # 力矩终止：原始PD力矩超过逐关节力矩上限的1.5倍
    torques_raw = mujoco_data.get("torques_raw", None)
    tau_limit = mujoco_data.get("tau_limit", None)
    if torques_raw is not None and tau_limit is not None:
        try:
            import numpy as _np
            tr = _np.asarray(torques_raw).reshape(-1)
            tl = _np.asarray(tau_limit).reshape(-1)
            over = _np.abs(tr) > (3 * tl)
            if _np.any(over):
                idx = _np.where(over)[0][:5]
                msg = ", ".join([f"j{int(i)}:{float(tr[i]):.1f}>(3*{float(tl[i]):.1f})" for i in idx])
                print(f"[Termination] Step {counter}: torque over-limit ({msg})")
                return True, "torque"
        except Exception:
            pass

    return False, None


def process_motion_data(input_path, cfg):
    """
    处理采集到的运动数据，分离episode并过滤短轨迹
    """
    print(f"\n=== 开始处理运动数据 ===")
    print(f"输入文件: {input_path}")

    # 读取原始数据
    data = joblib.load(input_path)
    # 兼容两种保存格式：
    # 1) 扁平旧格式：{ 'root_trans_offset': ..., 'terminate': ..., 'fps': ... }
    # 2) 包装新格式：{ 'motion0': { 'root_trans_offset': ..., 'terminate': ..., 'fps': ... }, ... }
    if isinstance(data, dict) and 'terminate' not in data:
        # 若顶层没有关键字段，尝试解包第一条 motion
        if len(data) > 0:
            first_key = next(iter(data))
            inner = data.get(first_key)
            if isinstance(inner, dict):
                data = inner
    motion = data
    # 基础校验：必须包含 terminate
    if isinstance(motion, dict) and 'terminate' not in motion:
        raise KeyError("Loaded motion does not contain 'terminate'. Please ensure the saved file structure is {'motion0': {..., 'terminate': [...]}} or flat dict with 'terminate'.")

    # 取出terminate信号
    terminate = np.array(motion['terminate'])
    T = len(terminate)

    print(f"总帧数: {T}")
    print(f"terminate标志数量: {np.sum(terminate)}")

    # 找到所有终止点
    episode_ends = np.where(terminate)[0]
    if len(episode_ends) == 0 or episode_ends[-1] != T - 1:
        episode_ends = np.append(episode_ends, T - 1)
    episode_starts = np.concatenate(([0], episode_ends[:-1] + 1))

    # print(f"Episode起始点: {episode_starts}")
    # print(f"Episode结束点: {episode_ends}")

    # 计算每个episode的长度
    episode_lengths = episode_ends - episode_starts + 1
    # print(f"Episode长度: {episode_lengths}")
    print(f"平均长度: {np.mean(episode_lengths):.1f}")
    # print(f"最短长度: {np.min(episode_lengths)}")
    # print(f"最长长度: {np.max(episode_lengths)}")

    # 切分每个episode
    episodes = []
    for i, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        ep = {}
        for k in motion.keys():
            if isinstance(motion[k], np.ndarray) and motion[k].shape[0] == T:
                ep[k] = motion[k][start:end + 1]
            else:
                ep[k] = motion[k]
        episodes.append(ep)
        # print(f"Episode {i}: {end - start + 1} 帧")

    print(f"\n共切分出{len(episodes)}条episode")

    # 过滤长度不足的episode
    MIN_LENGTH = cfg.min_episode_length
    filtered_episodes = []
    filtered_lengths = []

    print(f"\n=== 过滤长度{MIN_LENGTH}以下的episode ===")
    for i, (ep, length) in enumerate(zip(episodes, episode_lengths)):
        if length >= MIN_LENGTH:
            filtered_episodes.append(ep)
            filtered_lengths.append(length)
            print(f"保留 Episode {i}: {length} 帧")
        else:
            print(f"去除 Episode {i}: {length} 帧 (< {MIN_LENGTH})")

    removed_count = len(episodes) - len(filtered_episodes)
    print(f"\n📊 过滤统计:")
    print(f"原始episode数量: {len(episodes)}")
    print(f"去除的episode数量: {removed_count}")
    print(f"保留的episode数量: {len(filtered_episodes)}")

    if len(filtered_episodes) == 0:
        print("\n❌ 没有符合长度要求的episode!")
        return None

    # 转换数据格式
    filtered_data = {}
    for i, ep in enumerate(filtered_episodes):
        converted_ep = {}
        required_fields = ['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'action',
                           'terminate', 'root_lin_vel', 'root_ang_vel', 'dof_vel', 'motion_times']

        for key, value in ep.items():
            if key in required_fields:
                if isinstance(value, np.ndarray):
                    if key == 'action':
                        # 修复action维度：从(T,1,27)改为(T,27)
                        if len(value.shape) == 3 and value.shape[1] == 1:
                            converted_ep[key] = value.squeeze(1).astype(np.float32)
                        else:
                            converted_ep[key] = value.astype(np.float32)
                    elif key == 'terminate':
                        converted_ep[key] = value.astype(np.int64)
                    elif key in ['root_trans_offset', 'pose_aa', 'dof', 'root_rot',
                                 'root_lin_vel', 'root_ang_vel', 'dof_vel', 'motion_times']:
                        converted_ep[key] = value.astype(np.float32)
                    else:
                        converted_ep[key] = value
                else:
                    converted_ep[key] = value

        converted_ep['fps'] = 50.0
        filtered_data[f'motion{i}'] = converted_ep

    # 生成输出文件名
    output_dir = 'processed_pkl_files'
    os.makedirs(output_dir, exist_ok=True)

    input_basename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(input_basename)[0]

    if name_without_ext.startswith('2025'):
        name_without_year = name_without_ext[4:]
    else:
        name_without_year = name_without_ext

    output_filename = f'{name_without_year}_processed_min{MIN_LENGTH}.pkl'
    output_path = os.path.join(output_dir, output_filename)

    # 保存处理后的数据
    joblib.dump(filtered_data, output_path)

    filtered_lengths = np.array(filtered_lengths)
    print(f"\n✅ 过滤后的轨迹已保存为: {output_path}")
    print(f"📊 总帧数: {np.sum(filtered_lengths)}")
    print(f"⏱️ 总时长: {np.sum(filtered_lengths) * 0.02:.2f}秒")
    print(f"🔧 数据格式已匹配训练格式")

    return output_path


def run_and_save_mujoco(cfg, save_path):
    current_step = 0
    motions_for_saving = {'root_trans_offset': [], 'pose_aa': [], 'dof': [], 'root_rot': [], 'action': [],
                          'terminate': [], "root_lin_vel": [],
                          "root_ang_vel": [], "dof_vel": [], "motion_times": []}
    dt = cfg.simulation_dt * cfg.control_decimation
    dof_axis = parse_dof_axis_from_xml(cfg.xml_path)
    joint_names = parse_joint_names_from_xml(cfg.xml_path)
    torque_log = [] if cfg.log_torque else None  # list of (time, tau[27])
    # 详细日志：action_raw(未clip)、action_clip(clip后)、target_delta(缩放后)、tau_raw(PD未限幅)、tau(限幅后)
    detail_log = [] if cfg.log_torque else None
    # 统计力矩裁剪（饱和）比例
    sat_counts = np.zeros(cfg.num_actions, dtype=np.int64) if cfg.log_torque else None
    sat_samples = 0 if cfg.log_torque else None

    termination_stats = {
        'gravity_terminations': 0,
        'height_terminations': 0,
        'torque_terminations': 0,
        'normal_completions': 0,
        'total_episodes': 0
    }

    while True:
        # 回合初始化
        model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        data = mujoco.MjData(model)
        model.opt.timestep = cfg.simulation_dt

        # 显式设置基座位姿与关节初值，并清零速度
        try:
            data.qpos[:3] = cfg.init_base_pos  # base xyz
            data.qpos[3:7] = cfg.init_base_quat_wxyz  # base quat (w,x,y,z)
        except Exception:
            pass
        data.qpos[-cfg.num_actions:] = cfg.init_pos
        try:
            data.qvel[:] = 0.0
        except Exception:
            pass
        mujoco.mj_step(model, data)

        # 初始化
        model.opt.timestep = cfg.simulation_dt

        # mujoco可视化设置
        viewer = None
        if cfg.render:
            try:
                viewer = mujoco_viewer.MujocoViewer(model, data)
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -45
                viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
            except Exception as e:
                print(f"[Render] Disable rendering due to error: {e}")
                cfg.render = False

        # 策略模型加载
        onnx_model_path = cfg.policy_path
        policy = onnxruntime.InferenceSession(onnx_model_path)

        # 变量初始化
        # target_dof_pos = np.zeros((1, len(cfg.default_dof_pos.copy())))
        target_dof_pos = cfg.default_dof_pos.copy()
        action = np.zeros(cfg.num_actions, dtype=np.float32)

        # 初始化历史观测
        hist_dict = {'actions': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'base_ang_vel': np.zeros((cfg.frame_stack, 3), dtype=np.double),
                     'dof_pos': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'dof_vel': np.zeros((cfg.frame_stack, cfg.num_actions), dtype=np.double),
                     'projected_gravity': np.zeros((cfg.frame_stack, 3), dtype=np.double),
                     'ref_motion_phase': np.zeros((cfg.frame_stack, 1), dtype=np.double),
                     }
        history_keys = ['actions', 'base_ang_vel', 'dof_pos',
                        'dof_vel', 'projected_gravity', 'ref_motion_phase']
        hist_obs = []
        for key in history_keys:
            hist_obs.append(hist_dict[key].reshape(1, -1))
        hist_obs_c = np.concatenate(hist_obs, axis=1)
        counter = 0
        terminate_flag = False
        episode_terminated_early = False
        episode_terminated_early_reason = None

        last_tau = None
        for step in range(cfg.episode_steps * cfg.control_decimation):
            mujoco_data = get_mujoco_data(data)

            tau_raw = pd_control(target_dof_pos, mujoco_data["mujoco_dof_pos"],
                                 np.zeros_like(cfg.kds), mujoco_data["mujoco_dof_vel"], cfg)
            tau_limit = np.array(cfg.tau_limit)
            tau = np.clip(tau_raw, -tau_limit, tau_limit)
            mujoco_data['torques'] = tau
            last_tau = np.asarray(tau).astype(np.float32).reshape(-1)

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            if counter % cfg.control_decimation == 0:
                current_step += 1
                # ✅ 重新获取最新状态用于观测和早停检查
                mujoco_data = get_mujoco_data(data)
                # 提供原始未限幅力矩与力矩上限给早停判据
                mujoco_data['torques_raw'] = tau_raw
                mujoco_data['tau_limit'] = cfg.tau_limit
                # 记录本控制步的力矩与饱和
                if cfg.log_torque and last_tau is not None and torque_log is not None:
                    sim_time = counter * cfg.simulation_dt
                    tau_copy = np.asarray(last_tau).astype(np.float32).reshape(-1).copy()
                    torque_log.append((sim_time, tau_copy))
                    # 饱和统计：用未限幅 PD 力矩判断
                    tl = np.asarray(cfg.tau_limit).reshape(-1)
                    tau_raw_now = np.asarray(tau_raw).astype(np.float32).reshape(-1)
                    saturated = np.abs(tau_raw_now) > (tl + 1e-6)
                    sat_counts += saturated.astype(np.int64)
                    sat_samples += 1

                obs_buff, hist_obs_c = get_obs(hist_obs_c, hist_dict, mujoco_data, action, counter, cfg)
                policy_input = {policy.get_inputs()[0].name: obs_buff}
                action_raw = policy.run(["action"], policy_input)[0]
                action_clipped = np.clip(action_raw, -cfg.clip_actions, cfg.clip_actions)
                target_dof_pos = action_clipped * cfg.action_scale + cfg.default_dof_pos
                if cfg.log_torque and detail_log is not None:
                    target_delta = (target_dof_pos - cfg.default_dof_pos).astype(np.float32).reshape(-1)
                    detail_log.append(
                        (
                            sim_time,
                            np.asarray(action_raw).astype(np.float32).reshape(-1).copy(),
                            np.asarray(action_clipped).astype(np.float32).reshape(-1).copy(),
                            target_delta.copy(),
                            np.asarray(tau_raw).astype(np.float32).reshape(-1).copy(),
                            np.asarray(tau).astype(np.float32).reshape(-1).copy(),
                        )
                    )
                # 下一步循环使用 clip 后的动作作为观测中的 actions
                action = action_clipped

                should_terminate, term_reason = check_termination(mujoco_data, cfg, counter)

                # 保存数据
                q = data.qpos.astype(np.double)
                dq = data.qvel.astype(np.double)
                quat = np.array([q[4], q[5], q[6], q[3]])
                root_trans = q[:3]
                root_rot = quat
                dof = q[7:]
                root_rot_vec = R.from_quat(root_rot).as_rotvec()

                joint_aa = dof[:, None] * dof_axis
                # ATOM 有 5 个扩展关节：left_hand, right_hand, head, left_toe, right_toe
                num_augment_joint = 5
                pose_aa = np.concatenate([
                    root_rot_vec[None, :],
                    joint_aa,
                    np.zeros((num_augment_joint, 3), dtype=np.float32)
                ], axis=0)
                root_lin_vel = dq[:3]
                root_ang_vel = dq[3:6]
                dof_vel = dq[6:]

                if not cfg.render:
                    motions_for_saving['root_trans_offset'].append(root_trans)
                    motions_for_saving['root_rot'].append(root_rot)
                    motions_for_saving['dof'].append(dof)
                    motions_for_saving['pose_aa'].append(pose_aa)
                    motions_for_saving['action'].append(action)
                    motions_for_saving['root_lin_vel'].append(root_lin_vel)
                    motions_for_saving['root_ang_vel'].append(root_ang_vel)
                    motions_for_saving['dof_vel'].append(dof_vel)
                    motion_times = counter * cfg.simulation_dt
                    motions_for_saving['motion_times'].append(motion_times)
                    motions_for_saving['fps'] = 1.0 / dt

                    if should_terminate:
                        motions_for_saving['terminate'].append(True)
                        print(f"[Early Termination] Episode terminated at step {counter}/{cfg.episode_steps}")
                        episode_terminated_early = True
                        episode_terminated_early_reason = term_reason
                    elif ((current_step) % cfg.episode_steps) == 0:
                        motions_for_saving['terminate'].append(True)
                    else:
                        motions_for_saving['terminate'].append(False)

                    print(f"current_step:{current_step}/total_step:{cfg.total_steps}")

                if should_terminate:
                    break

            counter += 1

            render_freq = 50  # 30 FPS
            #render_interval = max(1, int(200 / render_freq))  # 每 6-7 步渲染一次
            render_interval = 5
            if cfg.render and viewer is not None and step % render_interval == 0:
                viewer.render()

        # 打印本回合最后一步逐关节力矩
        if cfg.log_torque and last_tau is not None:
            try:
                if len(joint_names) == len(last_tau):
                    print("\n[Final Joint Torques] (name: torque)")
                    for n, v in zip(joint_names, last_tau):
                        print(f"  {n}: {float(v):.4f}")
                else:
                    print("\n[Final Joint Torques]", last_tau)
            except Exception as e:
                print(f"[Warn] printing torques failed: {e}")

        # 保存全程力矩为 CSV 与可选 PNG
        try:
            if cfg.log_torque and torque_log is not None and len(torque_log) > 0:
                out_dir = os.path.join(os.getcwd(), "outputs")
                os.makedirs(out_dir, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(out_dir, f"torques_{ts}.csv")
                # 写表头: time + joint names
                with open(csv_path, "w") as f:
                    header = ["time"] + joint_names if len(joint_names) == torque_log[0][1].shape[0] else ["time"] + [
                        f"joint_{i}" for i in range(len(torque_log[0][1]))]
                    f.write(",".join(header) + "\n")
                    for t, tau in torque_log:
                        tau_flat = np.asarray(tau).reshape(-1)
                        row = [f"{float(t):.6f}"] + [f"{float(v):.6f}" for v in tau_flat]
                        f.write(",".join(row) + "\n")
                print(f"[Torque] Saved torque CSV: {csv_path}")

                # 输出饱和统计
                if sat_samples and sat_samples > 0:
                    rates = (sat_counts / float(sat_samples)).astype(np.float32)
                    print("\n被裁减率")
                    for n, r in zip(joint_names, rates):
                        print(f"  {n}: {r * 100:.1f}% ({int(sat_counts[joint_names.index(n)])}/{sat_samples})")
                    sat_csv_path = os.path.join(out_dir, f"torque_saturation_{ts}.csv")
                    with open(sat_csv_path, "w") as f:
                        f.write("joint,rate,count,samples,limit\n")
                        for i, n in enumerate(joint_names):
                            f.write(
                                f"{n},{rates[i]:.6f},{int(sat_counts[i])},{int(sat_samples)},{float(cfg.tau_limit[i]):.6f}\n")
                    print(f"[Torque] Saved saturation CSV: {sat_csv_path}")

                # 保存详细控制/力矩日志
                if detail_log is not None and len(detail_log) > 0:
                    det_csv_path = os.path.join(out_dir, f"controls_detail_{ts}.csv")
                    with open(det_csv_path, "w") as f:
                        header = ["time"]
                        for n in joint_names:
                            header.append(f"{n}:action_raw")
                        for n in joint_names:
                            header.append(f"{n}:action_clip")
                        for n in joint_names:
                            header.append(f"{n}:target_delta")
                        for n in joint_names:
                            header.append(f"{n}:tau_raw")
                        for n in joint_names:
                            header.append(f"{n}:tau")
                        f.write(",".join(header) + "\n")
                        for rec in detail_log:
                            t_rec, a_raw, a_clip, t_delta, tr_raw, tr = rec
                            row = [f"{float(t_rec):.6f}"]
                            row += [f"{float(v):.6f}" for v in a_raw]
                            row += [f"{float(v):.6f}" for v in a_clip]
                            row += [f"{float(v):.6f}" for v in t_delta]
                            row += [f"{float(v):.6f}" for v in tr_raw]
                            row += [f"{float(v):.6f}" for v in tr]
                            f.write(",".join(row) + "\n")
                    print(f"[Torque] Saved control detail CSV: {det_csv_path}")
                # 尝试保存 PNG（若 matplotlib 可用）
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    times = [float(t) for t, _ in torque_log]
                    taus = np.stack([np.asarray(tau).reshape(-1) for _, tau in torque_log], axis=0)
                    plt.figure(figsize=(12, 6))
                    for j in range(taus.shape[1]):
                        plt.plot(times, taus[:, j], linewidth=0.8)
                    plt.xlabel("time (s)")
                    plt.ylabel("torque (Nm)")
                    plt.title("Joint torques over time")
                    plt.grid(True, alpha=0.3)
                    png_path = os.path.join(out_dir, f"torques_{ts}.png")
                    plt.tight_layout()
                    plt.savefig(png_path, dpi=150)
                    plt.close()
                    print(f"[Torque] Saved torque plot: {png_path}")
                except Exception as e:
                    print(f"[Torque] Skip plotting (matplotlib not available?): {e}")
        except Exception as e:
            print(f"[Torque] Failed to save torque logs: {e}")

        # 更新统计信息
        termination_stats['total_episodes'] += 1
        if episode_terminated_early:
            if episode_terminated_early_reason == "gravity":
                termination_stats['gravity_terminations'] += 1
            elif episode_terminated_early_reason == "height":
                termination_stats['height_terminations'] += 1
            elif episode_terminated_early_reason == "torque":
                termination_stats['torque_terminations'] += 1
            else:
                # 未知原因归到 gravity 以保持兼容
                termination_stats['gravity_terminations'] += 1
        else:
            termination_stats['normal_completions'] += 1

        if current_step >= cfg.total_steps:
            break
        if cfg.render and viewer is not None:
            viewer.close()

    # 打印termination统计
    print(f"\n[Termination Statistics]")
    print(f"Total Episodes: {termination_stats['total_episodes']}")
    early_total = (termination_stats['gravity_terminations']
                   + termination_stats['height_terminations']
                   + termination_stats['torque_terminations'])
    print(f"Normal Completions: {termination_stats['normal_completions']}")
    print(f"Early Terminations: {early_total}")
    rate = (early_total / termination_stats['total_episodes'] * 100.0) if termination_stats[
                                                                              'total_episodes'] > 0 else 0.0
    print(f"Early Termination Rate: {rate:.1f}%")

    if not cfg.render:
        # 拼接所有list为ndarray
        result = {}
        for k in motions_for_saving:
            if k != 'fps':
                result[k] = np.array(motions_for_saving[k])
        result['fps'] = motions_for_saving['fps']

        # 包装成训练代码期望的格式: {'motion0': {...}}
        dump_data = {'motion0': result}

        # 保存原始数据
        save_f = open(save_path, 'wb')
        pickle.dump(dump_data, save_f)
        save_f.close()

        print(f"✅ Motion data saved to: {save_path}")
        print(f"📊 Total frames saved: {len(result['motion_times'])}")
        print(f"⏱️ Total duration: {result['motion_times'][-1]:.2f}s")
        print(f"🎬 FPS: {result['fps']:.1f}")

        return save_path

    return None


def main():
    """
    ATOM 机器人轨迹采集主函数
    """
    current_directory = os.getcwd()
    config_file = os.path.join(current_directory, "atom_config", "mujoco_config_atom.yaml")

    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        print("请先确保 atom_config/mujoco_config_atom.yaml 已正确配置")
        return

    print(f"📋 加载 ATOM 配置: {config_file}")
    cfg = read_conf(config_file)

    # 验证配置
    assert cfg.num_actions == 27, f"ATOM 应有 27 DOF，当前配置为 {cfg.num_actions}"
    assert len(cfg.kps) == 27, f"kps 应有 27 个值，当前有 {len(cfg.kps)}"
    assert len(cfg.kds) == 27, f"kds 应有 27 个值，当前有 {len(cfg.kds)}"
    assert cfg.num_single_obs == 88, f"单帧观测应为 88 维，当前为 {cfg.num_single_obs}"

    print("✅ 配置验证通过")
    print(f"  - DOF: {cfg.num_actions}")
    print(f"  - 单帧观测: {cfg.num_single_obs}")
    print(f"  - 完整观测: 440 (含历史)")
    print(f"  - Episode 步数: {cfg.episode_steps}")
    print(f"  - 总步数: {cfg.total_steps}")

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_save_path = os.path.join(current_directory, f"{timestamp}_atom_motion_raw.pkl")

    print(f"\n=== 开始 ATOM 轨迹采集 ===")
    print(f"原始数据将保存到: {raw_save_path}")

    # 步骤1：运行仿真并保存原始数据
    saved_path = run_and_save_mujoco(cfg, raw_save_path)

    if saved_path and cfg.auto_process:
        print(f"\n=== 开始自动处理数据 ===")
        # 步骤2：自动处理数据
        processed_path = process_motion_data(saved_path, cfg)

        if processed_path:
            print(f"\n🎉 ATOM 轨迹采集完成!")
            print(f"📁 原始数据: {saved_path}")
            print(f"📁 处理后数据: {processed_path}")
            print(f"\n💡 处理后的数据可直接用于训练，格式已匹配 humanoidverse")
            print(f"   观测维度: 440 (actor_obs)")
            print(f"   动作维度: 27 (ATOM DOF)")

            # 可选：删除原始文件以节省空间
            # os.remove(saved_path)
            # print(f"🗑️ 原始文件已删除")
        else:
            print(f"\n⚠️ 数据处理失败，原始数据保留在: {saved_path}")
    else:
        print(f"\n✅ ATOM 轨迹采集完成，数据保存在: {saved_path}")
        if not cfg.auto_process:
            print("💡 如需处理数据，请在配置文件中设置 auto_process: true")

    print("\n" + "=" * 50)
    print("✅ 任务完成")
    print("=" * 50)


if __name__ == '__main__':
    main()