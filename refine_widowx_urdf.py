import os
import argparse
import xml.etree.ElementTree as ET
import trimesh

def scale_and_save_mesh(original_filepath, output_folder, scale_factor=1000.0):
    """
    读取原始 mesh 文件，对其顶点进行缩放（除以 scale_factor），并保存到 output_folder 中。
    返回保存后的新文件路径。
    """
    if not os.path.exists(original_filepath):
        raise FileNotFoundError(f"文件不存在: {original_filepath}")

    # 加载 mesh 文件（支持 obj、stl 等格式）
    mesh = trimesh.load(original_filepath)
    # 如果加载结果为 Scene，则合并所有几何体
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces) 
             for geom in mesh.geometry.values()]
        )
    # 对所有顶点进行缩放
    mesh.vertices /= scale_factor

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    basename = os.path.basename(original_filepath)
    new_filepath = os.path.join(output_folder, basename)
    
    # 导出 mesh，新文件格式与原格式一致
    mesh.export(new_filepath)
    print(f"已生成缩放后的 mesh: {original_filepath} -> {new_filepath}")
    return new_filepath

def process_urdf(input_urdf, output_urdf, mesh_output_folder, scale_factor=1000.0):
    """
    读取 URDF 文件，对其中每个引用的 mesh 文件（.obj 和 .stl）进行缩放，
    将缩放后的文件保存到 mesh_output_folder 中，并更新 URDF 文件中对应的引用路径，
    同时将 scale 属性更新为 "1 1 1"。
    """
    tree = ET.parse(input_urdf)
    root = tree.getroot()

    for mesh in root.iter('mesh'):
        filename_attr = mesh.get('filename')
        if filename_attr and (filename_attr.lower().endswith('.obj') or filename_attr.lower().endswith('.stl')):
            original_mesh_path = os.path.join(os.path.dirname(input_urdf), filename_attr)
            try:
                new_mesh_path = scale_and_save_mesh(original_mesh_path, mesh_output_folder, scale_factor)
            except Exception as e:
                print(f"处理文件 {original_mesh_path} 时出错: {e}")
                continue

            new_rel_path = os.path.relpath(new_mesh_path, os.path.dirname(output_urdf))
            mesh.set('filename', new_rel_path)
            mesh.set('scale', "1 1 1")
            print(f"更新 URDF 中文件引用: {filename_attr} -> {new_rel_path}")

    tree.write(output_urdf, encoding='utf-8', xml_declaration=True)
    print(f"修改后的 URDF 文件已保存到: {output_urdf}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="对 URDF 文件内引用的 mesh 文件进行缩放（单位缩小 1000 倍），并将缩放后的文件保存到指定文件夹，同时更新 URDF 文件中对应的引用路径。"
    )
    parser.add_argument("input_urdf", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/wx250s.urdf", help="输入的 URDF 文件路径")
    parser.add_argument("output_urdf", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/scale_wx250s.urdf", help="输出修改后的 URDF 文件路径")
    parser.add_argument("mesh_output_folder", nargs="?", default="/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description/scale_wx250s", help="缩放后的 mesh 文件保存的新文件夹")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="缩放因子（默认 1000）")
    args = parser.parse_args()

    process_urdf(args.input_urdf, args.output_urdf, args.mesh_output_folder, args.scale_factor)
