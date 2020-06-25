from lxml import etree


def write_vxd(genotype, seed, iid):

    (x, y, z) = genotype.orig_size_xyz

    root = etree.Element("VXD")  # new vxd root

    structure = etree.SubElement(root, "Structure")
    structure.set('replace', 'VXA.VXC.Structure')
    structure.set('Compression', 'ASCII_READABLE')
    etree.SubElement(structure, "X_Voxels").text = str(x)
    etree.SubElement(structure, "Y_Voxels").text = str(y)
    etree.SubElement(structure, "Z_Voxels").text = str(z)

    for name, details in genotype.to_phenotype_mapping.items():
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
    with open('data'+str(seed)+'/bot_{:04d}.vxd'.format(iid), 'wb') as vxd:
        vxd.write(etree.tostring(root))
