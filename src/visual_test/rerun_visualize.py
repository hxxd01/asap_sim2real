import argparse
import time
import numpy as np
import pinocchio as pin
import rerun as rr
import trimesh
from scipy.spatial.transform import Rotation as R

class RerunURDF():
    def __init__(self):
        self.name = "atom"
        self.robot = pin.RobotWrapper.BuildFromURDF('assets/atom_description/atom_fixed_upper.urdf', 'assets/atom_description/', pin.JointModelFreeFlyer())
        self.Tpose = np.array([0,0,0.96,0,0,0,1,
                                -0.15,0,0,0.3,-0.15,0,
                                -0.15,0,0,0.3,-0.15,0]).astype(np.float32)
        
        # print all joints names
        for i in range(self.robot.model.njoints):
            print(self.robot.model.names[i])
        
        self.link2mesh = self.get_link2mesh()
        self.load_visual_mesh()
        self.update()
    
    def get_link2mesh(self):
        link2mesh = {}
        for visual in self.robot.visual_model.geometryObjects:
            mesh = trimesh.load_mesh(visual.meshPath)
            name = visual.name[:-2]
            mesh.visual = trimesh.visual.ColorVisuals()
            mesh.visual.vertex_colors = visual.meshColor
            link2mesh[name] = mesh
        return link2mesh
   
    def load_visual_mesh(self):       
        self.robot.framesForwardKinematics(pin.neutral(self.robot.model))
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            mesh = self.link2mesh[frame_name]
            
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            frame_tf = self.robot.data.oMf[frame_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
            
            relative_tf = joint_tf.inverse() * frame_tf
            mesh.apply_transform(relative_tf.homogeneous)
            rr.log(f'urdf_{self.name}/{parent_joint_name}/{frame_name}',
                   rr.Mesh3D(
                       vertex_positions=mesh.vertices,
                       triangle_indices=mesh.faces,
                       vertex_normals=mesh.vertex_normals,
                       vertex_colors=mesh.visual.vertex_colors,
                       albedo_texture=None,
                       vertex_texcoords=None,
                   ),
                   static=True)
    
    def update(self, configuration = None):        
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parent
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            rr.log(f'urdf_{self.name}/{parent_joint_name}',
                   rr.Transform3D(translation=joint_tf.translation,
                                  mat3x3=joint_tf.rotation,
                                  axis_length=0.01))
    def convert_euler_to_quaternion(self, zyx):
            # Convert ZYX Euler angles to quaternion
            rotation = R.from_euler('zyx', zyx, degrees=False)
            return rotation.as_quat()
    def create_configuration(self, row):
        x, y, z, r, p, yaw, *qpos = row
        quaternion = self.convert_euler_to_quaternion([r, p, yaw])
        # Format: x, y, z, qw, qx, qy, qz, qpos
        configuration = np.hstack([x, y, z, quaternion, qpos])
        return configuration
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject2')
    args = parser.parse_args()

    rr.init(
        'Reviz', 
        spawn=True
    )
    rr.log('', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    file_name = args.file_name
    csv_files =  'policy/atom/' + file_name + '.csv'
    data = np.genfromtxt(csv_files, delimiter=',', skip_header=1, usecols=range(0, 18))
    
    print("data.shape[0]: ", data.shape[0])
    print("data.shape: ", data.shape)

    print(data[:5, :])  # Print the first 5 rows of data

    rerun_urdf = RerunURDF()
    for frame_nr in range(data.shape[0]):
        rr.set_time_sequence('frame_nr', frame_nr)
        row = data[frame_nr, :] 
        # print(len(row))
        # print('##############')
        # print(row)
        # print('#######222#######')

        configuration = rerun_urdf.create_configuration(row)
        rerun_urdf.update(configuration)
        time.sleep(0.005)
