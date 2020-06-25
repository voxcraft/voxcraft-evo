import random
from copy import deepcopy
from scipy.ndimage.measurements import label

import numpy as np

from evosorocore.NetworkUtils import sigmoid
from evosorocore.Networks import Network


class Genotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    def __init__(self, orig_size_xyz=(6, 6, 6)):

        """
        Parameters
        ----------
        orig_size_xyz : 3-tuple (x, y, z)
            Defines the original 3 dimensions for the cube of voxels corresponding to possible networks outputs. The
            maximum number of SofBot voxel components is x*y*z, a full cube.

        """
        self.networks = []
        self.all_networks_outputs = []
        self.to_phenotype_mapping = GenotypeToPhenotypeMap()
        self.orig_size_xyz = orig_size_xyz

    def __iter__(self):
        """Iterate over the networks. Use the expression 'for n in network'."""
        return iter(self.networks)

    def __len__(self):
        """Return the number of networks in the genotype. Use the expression 'len(network)'."""
        return len(self.networks)

    def __getitem__(self, n):
        """Return network n.  Use the expression 'network[n]'."""
        return self.networks[n]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        return "Genotype made of: " + ',\n'.join(map(str, self.networks))

    def mutate(self, **kwargs):
        """
        Selects one network at random and attempts to mutate it.
        :param kwargs: Not currently used.
        :return: None
        """
        random.choice(self.networks).mutate()

    def add_network(self, network):
        """Append a new network to this list of networks.

        Parameters
        ----------
        network : Network
            The network to add. Should be a subclass of Network.

        """
        assert isinstance(network, Network)
        self.networks += [network]
        self.all_networks_outputs.extend(network.output_node_names)

    def express(self):
        """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""

        for network in self:
            for name in network.graph.nodes():
                network.graph.nodes[name]["evaluated"] = False  # flag all nodes as unevaluated

            network.set_input_node_states(self.orig_size_xyz)  # reset the inputs

            for name in network.output_node_names:
                network.graph.nodes[name]["state"] = np.zeros(self.orig_size_xyz)  # clear old outputs
                network.graph.nodes[name]["state"] = self.calc_node_state(network, name)  # calculate new outputs

        for network in self:
            for name in network.output_node_names:
                if name in self.to_phenotype_mapping:
                    self.to_phenotype_mapping[name]["state"] = network.graph.nodes[name]["state"]

        for name, details in self.to_phenotype_mapping.items():
            if name not in self.all_networks_outputs:
                details["state"] = np.ones(self.orig_size_xyz, dtype=details["output_type"]) * -999
                if details["dependency_order"] is not None:
                    for dependency_name in details["dependency_order"]:
                        self.to_phenotype_mapping.dependencies[dependency_name]["state"] = None

        for name, details in self.to_phenotype_mapping.items():
            if details["dependency_order"] is not None:
                details["state"] = details["func"](self)

    def calc_node_state(self, network, node_name):
        """Propagate input values through the network"""
        if network.graph.nodes[node_name]["evaluated"]:
            return network.graph.nodes[node_name]["state"]

        network.graph.nodes[node_name]["evaluated"] = True
        input_edges = network.graph.in_edges(nbunch=[node_name])
        new_state = np.zeros(self.orig_size_xyz)

        for edge in input_edges:
            node1, node2 = edge
            new_state += self.calc_node_state(network, node1) * network.graph.edges[node1, node2]["weight"]

        network.graph.nodes[node_name]["state"] = new_state

        if node_name in self.to_phenotype_mapping:
            if self.to_phenotype_mapping[node_name]["dependency_order"] is None:
                return self.to_phenotype_mapping[node_name]["func"](new_state)

        return network.graph.nodes[node_name]["function"](new_state)


class GenotypeToPhenotypeMap(object):
    """A mapping of the relationship from genotype (networks) to phenotype (VoxCad simulation)."""

    # TODO: generalize dependencies from boolean to any operation (e.g. to set an env param from multiple outputs)

    def __init__(self):
        self.mapping = dict()
        self.dependencies = dict()

    def items(self):
        """to_phenotype_mapping.items() -> list of (key, value) pairs in mapping"""
        return [(key, self.mapping[key]) for key in self.mapping]

    def __contains__(self, key):
        """Return True if key is a key str in the mapping, False otherwise. Use the expression 'key in mapping'."""
        try:
            return key in self.mapping
        except TypeError:
            return False

    def __len__(self):
        """Return the number of mappings. Use the expression 'len(mapping)'."""
        return len(self.mapping)

    def __getitem__(self, key):
        """Return mapping for node with name 'key'.  Use the expression 'mapping[key]'."""
        return self.mapping[key]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def add_map(self, name, tag, func=sigmoid, output_type=float, dependency_order=None):
        """Add an association between a genotype output and a VoxCad parameter.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        tag : str
            The tag used in parsing the resulting output from a VoxCad simulation.
            If this is None then the attribute is calculated outside of VoxCad (in Python only).

        func : func
            Specifies relationship between attributes and xml tag.

        output_type : type
            The output type

        dependency_order : list
            Order of operations

        """
        if (dependency_order is not None) and not isinstance(dependency_order, list):
            dependency_order = [dependency_order]

        self.mapping[name] = {"tag": tag,
                              "func": func,
                              "dependency_order": dependency_order,
                              "state": None,
                              "old_state": None,
                              "output_type": output_type
                              }

    def add_output_dependency(self, name, dependency_name, requirement, material_if_true=None, material_if_false=None):
        """Add a dependency between two genotype outputs.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        dependency_name : str
            Another network output node name.

        requirement : bool
            Dependency must be this

        material_if_true : int
            The material if dependency meets pre-requisite

        material_if_false : int
            The material otherwise

        """
        self.dependencies[name] = {"depends_on": dependency_name,
                                   "requirement": requirement,
                                   "material_if_true": material_if_true,
                                   "material_if_false": material_if_false,
                                   "state": None}

    def get_dependency(self, name, output_bool):
        """Checks recursively if all boolean requirements were met in dependent outputs."""
        if self.dependencies[name]["depends_on"] is not None:
            dependency = self.dependencies[name]["depends_on"]
            requirement = self.dependencies[name]["requirement"]
            return np.logical_and(self.get_dependency(dependency, True) == requirement,
                                  self.dependencies[name]["state"] == output_bool)
        else:
            return self.dependencies[name]["state"] == output_bool


class Phenotype(object):
    """Physical manifestation of the genotype - determines the physiology of an individual."""

    def __init__(self, genotype):

        """
        Parameters
        ----------
        genotype : Genotype cls
            Defines particular networks (the genome).

        """
        self.genotype = genotype()
        self.genotype.express()

        while not self.is_valid():
            self.genotype = genotype()
            self.genotype.express()

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        return '\n'.join([str(name) + ": " + str(details["state"]) for (name, details) in self.genotype.to_phenotype_mapping.items()])

    def get_phenotype(self):
        return self.genotype.to_phenotype_mapping.items()

    def mutate(self, max_mutation_attempts=1000):
        """
        Mutates the genotype and ensures that the mutation results in a valid phenotype.
        TODO: Do we want to ensure that the mutated genome is actually different from the main genome?
        TODO: ^yes
        :param max_mutation_attempts: Try to find a valid mutation up to this many times.
        :return: True if mutation occured, false otherwise.
        """
        old_genotype = self.genotype
        for i in range(max_mutation_attempts):
            self.genotype = deepcopy(old_genotype)
            self.genotype.mutate(max_mutation_attempts=max_mutation_attempts)
            self.genotype.express()
            if self.is_valid():
                return True
        return False

    def is_valid(self):
        """Ensures a randomly generated phenotype is valid (checked before adding individual to a population).

        Returns
        -------
        is_valid : bool
        True if self is valid, False otherwise.

        """
        for network in self.genotype:
            for output_node_name in network.output_node_names:
                if np.isnan(network.graph.nodes[output_node_name]["state"]).any():
                    return False

        return True


def make_material_tree(this_softbot, *args, **kwargs):
    mapping = this_softbot.to_phenotype_mapping
    material = mapping["material"]

    if material["dependency_order"] is not None:
        for dependency_name in material["dependency_order"]:
            for network in this_softbot:
                if dependency_name in network.graph.nodes():
                    mapping.dependencies[dependency_name]["state"] = network.graph.nodes[dependency_name]["state"] > 0

    if material["dependency_order"] is not None:
        for dependency_name in reversed(material["dependency_order"]):
            if mapping.dependencies[dependency_name]["material_if_true"] is not None:
                material["state"][mapping.get_dependency(dependency_name, True)] = \
                    mapping.dependencies[dependency_name]["material_if_true"]

            if mapping.dependencies[dependency_name]["material_if_false"] is not None:
                material["state"][mapping.get_dependency(dependency_name, False)] = \
                    mapping.dependencies[dependency_name]["material_if_false"]
    return make_one_shape_only(material["state"]) * material["state"]


def make_one_shape_only(output_state):
    """Find the largest continuous arrangement of True elements after applying boolean mask.

    Avoids multiple disconnected softbots in simulation counted as a single individual.

    Parameters
    ----------
    output_state : numpy.ndarray
        Network output

    Returns
    -------
    part_of_ind : bool
        True if component of individual

    """
    if np.sum(output_state) == 0:
        return output_state

    # find coordinates
    array = output_state > 0
    labeled, ncomponents = label(array)

    largest_count = 0
    largest_label = 0
    for n in range(ncomponents+1):
        this_count = np.sum(labeled == n)
        vox_count = np.sum(array[labeled == n])
        if (this_count > largest_count) and (vox_count > 0):
            largest_label = n
            largest_count = this_count

    return labeled == largest_label
