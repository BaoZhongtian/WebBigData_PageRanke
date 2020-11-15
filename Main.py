import os
import time
import numpy
import argparse
import matplotlib.pylab as plt


def main():
    startTime = time.time()
    print('Start Time =', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(startTime)))
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_path", default=None, type=str, required=True, help="The path of input file.")
    ## Default Page Rank parameters
    parser.add_argument("--output_path", default='Result.txt', type=str, required=False, help="The path of result.")
    parser.add_argument("--teleport_parameter", default=0.85, type=float, required=False,
                        help="The teleport parameter in the Page Rank Algorithm."
                             " If this parameter more than 1 or less than 0, that is meaning this is not used.")
    parser.add_argument("--not_dead_end_flag", default=False, action='store_true', required=False,
                        help="The dead end flag decide to use or not use method to treat with dead end situation in the DENSE METHOD.")
    parser.add_argument("--min_changes", default=1E-4, type=float, required=False,
                        help="The min changes between two epoch. Small this value will end the iteration.")
    parser.add_argument("--max_iteration_times", default=1E+6, type=int, required=False,
                        help="The maximum times for page rank iteration.")
    parser.add_argument("--top_node_number", default=100, type=int, required=False,
                        help="The top node number write to the output file.")
    ## Dense Matrix parameters
    parser.add_argument("--dense_flag", default=False, action='store_true', required=False,
                        help="The dense flag decide to use or not use dense matrix treatment.")
    parser.add_argument("--power_flag", default=False, action='store_true', required=False,
                        help="The power flag is to only calculate M1, M2, M4, M8 such the matrix."
                             "If this flag is on, it may be more quick to covergence."
                             "This Flag is only effective when the dense flag is TRUE.")

    ## Inner Test, Ignore it Plz#####################################################################
    args = parser.parse_args('--input_path=WikiData.txt'.split())
    # print(args.not_dead_end_flag)
    # exit()

    ##########################################################################################

    ## Check the parameter
    if not os.path.exists(args.input_path): raise RuntimeError('Can not find file: %s' % args.input_path)
    rawData = None
    for encoding in ['UTF-8', 'GBK']:
        try:
            with open(args.input_path, 'r', encoding=encoding) as file:
                rawData = file.readlines()
        except:
            pass
        if rawData is not None: break
    if rawData is None: raise RuntimeError('Can not decode the files: %s' % args.input_path)

    nodeDictionary = {}
    deadNodeSet = set()
    for index in range(len(rawData)):
        lineData = rawData[index].split('\t')
        if len(lineData) == 1 or len(lineData) >= 3:
            raise RuntimeError('Illegal Input in line %d\n%s' % (index, rawData[index]))
        lineData = [int(v) for v in lineData]
        deadNodeSet.add(lineData[0])
        deadNodeSet.add(lineData[1])

        if lineData[0] not in nodeDictionary:
            nodeDictionary[lineData[0]] = [lineData[1]]
        else:
            nodeDictionary[lineData[0]].append(lineData[1])
    for sample in nodeDictionary.keys():
        if sample in deadNodeSet: deadNodeSet.remove(sample)

    if args.dense_flag:
        __DensePageRank(nodeDictionary=nodeDictionary, deadNodeSet=deadNodeSet, args=args)
    else:
        __SparsePageRank(nodeDictionary=nodeDictionary, deadNodeSet=deadNodeSet, args=args)

    endTime = time.time()
    print('End Time  =', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(endTime)))
    print('It cost %.2f seconds.' % (endTime - startTime))
    plt.show()


def __DensePageRank(nodeDictionary, deadNodeSet, args):
    # Generate the Matrix M
    node2Index, index2Node = {}, {}
    for sample in nodeDictionary.keys():
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample
    for sample in deadNodeSet:
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample
    matrixM = numpy.zeros([len(node2Index.keys()), len(node2Index.keys())])
    for startNode in nodeDictionary.keys():
        for endNode in nodeDictionary[startNode]:
            matrixM[node2Index[endNode]][node2Index[startNode]] += 1.0 / len(nodeDictionary[startNode])

    if not args.not_dead_end_flag:
        # Dead End Consideration
        for sample in deadNodeSet:
            matrixM[:, node2Index[sample]] = 1.0 / len(node2Index.keys())

    if 0 < args.teleport_parameter <= 1:
        # Teleport Consideration
        supplementMatrix = numpy.ones([len(node2Index.keys()), len(node2Index.keys())]) / len(node2Index.keys())
        matrixM = matrixM * args.teleport_parameter + supplementMatrix * (1 - args.teleport_parameter)

    matrixMK = matrixM.copy()
    iterationTimes = 1
    nodeWeightPast = numpy.average(matrixMK, axis=1)
    iterationChanges = []
    while True:
        print('Iteration %d' % iterationTimes)
        if args.power_flag:
            iterationTimes *= 2
            matrixMK = numpy.matmul(matrixMK, matrixMK)
        else:
            iterationTimes += 1
            matrixMK = numpy.matmul(matrixMK, matrixM)
        nodeWeight = numpy.average(matrixMK, axis=1)

        iterationChanges.append([iterationTimes, numpy.sum(numpy.abs(nodeWeight - nodeWeightPast))])
        if numpy.sum(numpy.abs(nodeWeight - nodeWeightPast)) < args.min_changes:
            break
        else:
            nodeWeightPast = nodeWeight
        if iterationTimes > args.max_iteration_times:
            # raise RuntimeWarning('Can not convergence with assigned min change distance.')
            print('ERROR : Can not convergence with assigned min change distance.')
            break

    iterationChanges = numpy.array(iterationChanges)
    plt.plot(iterationChanges[:, 0], iterationChanges[:, 1])
    plt.xlabel('Iteration Times')
    plt.ylabel('Changes between iterations')

    nodeWeight = numpy.array(nodeWeight)
    nodeWeight = numpy.concatenate([nodeWeight[:, numpy.newaxis], numpy.arange(len(nodeWeight))[:, numpy.newaxis]],
                                   axis=1)
    nodeWeight = sorted(nodeWeight, key=lambda x: x[0], reverse=True)
    with open(args.output_path, 'w') as file:
        for nodeSample in nodeWeight[0:args.top_node_number]:
            file.write('%d %.20f\n' % (index2Node[nodeSample[1]], nodeSample[0]))


def __SparsePageRank(nodeDictionary, deadNodeSet, args):
    node2Index, index2Node = {}, {}
    for sample in nodeDictionary.keys():
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample
    for sample in deadNodeSet:
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample

    nodeWeightPast = numpy.ones(len(node2Index.keys())) / len(node2Index.keys())
    iterationTimes = 1
    iterationChanges = []

    while True:
        print('Iteration %d' % iterationTimes)
        nodeWeight = numpy.zeros(len(node2Index.keys()))
        for linkStart in nodeDictionary.keys():
            if 0 < args.teleport_parameter <= 1:
                nodeWeight += nodeWeightPast[node2Index[linkStart]] * (1 - args.teleport_parameter) / len(
                    node2Index.keys())
                for linkEnd in nodeDictionary[linkStart]:
                    nodeWeight[node2Index[linkEnd]] += nodeWeightPast[node2Index[linkStart]] / len(
                        nodeDictionary[linkStart]) * args.teleport_parameter
            else:
                for linkEnd in nodeDictionary[linkStart]:
                    nodeWeight[node2Index[linkEnd]] += nodeWeightPast[node2Index[linkStart]] / len(
                        nodeDictionary[linkStart])
        if not args.not_dead_end_flag:
            nodeWeight += (1 - numpy.sum(nodeWeight)) / len(node2Index.keys())

        iterationChanges.append([iterationTimes, numpy.sum(numpy.abs(nodeWeight - nodeWeightPast))])

        if numpy.sum(numpy.abs(nodeWeight - nodeWeightPast)) < args.min_changes:
            break
        else:
            nodeWeightPast = nodeWeight

        iterationTimes += 1
        if iterationTimes > args.max_iteration_times:
            # raise RuntimeWarning('Can not convergence with assigned min change distance.')
            print('ERROR : Can not convergence with assigned min change distance.')
            break

    iterationChanges = numpy.array(iterationChanges)
    plt.plot(iterationChanges[:, 0], iterationChanges[:, 1])
    plt.xlabel('Iteration Times')
    plt.ylabel('Changes between iterations')

    nodeWeight = numpy.array(nodeWeight)
    nodeWeight = numpy.concatenate([nodeWeight[:, numpy.newaxis], numpy.arange(len(nodeWeight))[:, numpy.newaxis]],
                                   axis=1)
    nodeWeight = sorted(nodeWeight, key=lambda x: x[0], reverse=True)
    with open(args.output_path, 'w') as file:
        for nodeSample in nodeWeight[0:args.top_node_number]:
            file.write('%d %.20f\n' % (index2Node[nodeSample[1]], nodeSample[0]))


if __name__ == '__main__':
    main()
