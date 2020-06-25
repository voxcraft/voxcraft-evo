import os
import subprocess as sub
from lxml import etree


def write_vxd(phenotype, data_folder_name, robot_id):

    (x, y, z) = phenotype.genotype.orig_size_xyz

    root = etree.Element("VXD")  # new vxd root

    structure = etree.SubElement(root, "Structure")
    structure.set('replace', 'VXA.VXC.Structure')
    structure.set('Compression', 'ASCII_READABLE')
    etree.SubElement(structure, "X_Voxels").text = str(x)
    etree.SubElement(structure, "Y_Voxels").text = str(y)
    etree.SubElement(structure, "Z_Voxels").text = str(z)

    for name, details in phenotype.get_phenotype():
        state = details["state"]
        flattened_state = state.reshape(z, x*y)

        nickname = name
        if name == "material":  # this is the keyword for voxel material type in the CPPN code
            nickname = "Data"  # this is the keyword expected by voxcraft-sim

        data = etree.SubElement(structure, nickname)
        for i in range(flattened_state.shape[0]):
            layer = etree.SubElement(data, "Layer")
            if nickname == "Data":
                str_layer = "".join([str(c) for c in flattened_state[i]])
            else:
                str_layer = "".join([str(c) + ", " for c in flattened_state[i]])  # comma separated

            layer.text = etree.CDATA(str_layer)

    # save the vxd to data folder
    with open('data_' + data_folder_name + '/bot_{:04d}.vxd'.format(robot_id), 'wb') as vxd:
        vxd.write(etree.tostring(root))


def remove_vxd(data_folder_name, robot_id):
    os.remove('data_' + data_folder_name + '/bot_{:04d}.vxd'.format(robot_id))


def read_vxd(data_folder_name, robot_id, fitness_tag):

    while True:
        try:
            sub.call("./voxcraft-sim -i data_{0} -o output{1}.xml".format(data_folder_name, robot_id), shell=True)
            # sub.call waits for the process to return
            # after it does, we collect the results output by the simulator
            # root = etree.parse("output{}.xml".format(seed)).getroot()
            break

        except IOError:
            print("Shoot! There was an IOError. I'll re-simulate this batch again...")
            pass

        except IndexError:
            print("Dang it! There was an IndexError. I'll re-simulate this batch again...")
            pass

    root = etree.parse("output{}.xml".format(robot_id)).getroot()
    fitness = float(root.findall("detail/bot_{:04d}/".format(robot_id) + fitness_tag)[0].text)

    return fitness
