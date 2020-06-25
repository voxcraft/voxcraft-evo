from lxml import etree
import subprocess as sub


def setup_folders(seed):
    sub.call("mkdir data{}".format(seed), shell=True)
    sub.call("cp base.vxa data{}/base.vxa".format(seed), shell=True)  # copy the base.vxa into the data folder


def evaluate(seed, iid, tag="fitness_score"):
    """
    Sends all of the vxd files in the data folder to voxcraft-sim.
    Reads all the output.xml files produced by the simulator.

    :param seed:
    :param iid: individual id
    :param tag: what xml tag we take to be the fitness

    """

    # clear any old .vxd robot files from the data directory
    sub.call("rm data{}/*.vxd".format(seed), shell=True)

    # remove any old sim output.xml
    sub.call("rm output{}.xml".format(seed), shell=True)

    while True:
        try:
            sub.call("./voxcraft-sim -i data{0} -o output{1}.xml".format(seed, seed), shell=True)
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

    root = etree.parse("output{}.xml".format(seed)).getroot()
    fitness = float(root.findall("detail/bot_{:04d}/".format(iid) + tag)[0].text)

    return fitness


