import os
import time
import numpy
import argparse


def PageRankMain(command=None):
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
    parser.add_argument("--min_changes", default=1E-3, type=float, required=False,
                        help="The min changes between two epoch. Small this value will end the iteration.")
    parser.add_argument("--max_iteration_times", default=1E+3, type=int, required=False,
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
    ## Sparse Matrix parameters
    parser.add_argument("--block_flag", default=False, action='store_true', required=False,
                        help="The block flag decide to use or not use block sparse matrix treatment.")
    parser.add_argument("--block_length", default=1000, type=int, required=False,
                        help="############################")

    if command is not None:
        args = parser.parse_args(command.split())
    else:
        args = parser.parse_args()

    ##########################################################################################

    ## Check the parameter
    args.input_path = args.input_path.replace('"', '')
    args.output_path = args.output_path.replace('"', '')
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
    print('Total Node Number :', len(deadNodeSet) + len(nodeDictionary.keys()))
    print('Total Line Number :', numpy.sum(len(nodeDictionary[v]) for v in nodeDictionary))
    print('Dead End Node Number :', len(deadNodeSet))

    if args.block_flag:
        iterationChanges, nodeWeight, index2Node = __SparseBlockPageRank(rawData, args)
    else:
        if args.dense_flag:
            iterationChanges, nodeWeight, index2Node = \
                __DensePageRank(nodeDictionary=nodeDictionary, deadNodeSet=deadNodeSet, args=args)
        else:
            iterationChanges, nodeWeight, index2Node = \
                __SparsePageRank(nodeDictionary=nodeDictionary, deadNodeSet=deadNodeSet, args=args)

    iterationChanges = numpy.array(iterationChanges)
    with open(os.path.join(args.output_path, 'DistanceRecord.txt'), 'w') as file:
        for sample in iterationChanges:
            file.write('%d %.20f\n' % (sample[0], sample[1]))

    nodeWeight = numpy.array(nodeWeight)
    nodeWeight = numpy.concatenate([nodeWeight[:, numpy.newaxis], numpy.arange(len(nodeWeight))[:, numpy.newaxis]],
                                   axis=1)
    nodeWeight = sorted(nodeWeight, key=lambda x: x[0], reverse=True)
    with open(os.path.join(args.output_path, 'Result.txt'), 'w') as file:
        for nodeSample in nodeWeight[0:args.top_node_number]:
            file.write('%d %.20f\n' % (index2Node[nodeSample[1]], nodeSample[0]))

    endTime = time.time()
    print('End Time  =', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(endTime)))
    print('It cost %.2f seconds.' % (endTime - startTime))
    # plt.show()


def __NodeIndexMatch(nodeDictionary, deadNodeSet):
    node2Index, index2Node = {}, {}
    for sample in nodeDictionary.keys():
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample
    for sample in deadNodeSet:
        node2Index[sample] = len(node2Index.keys())
        index2Node[node2Index[sample]] = sample
    return node2Index, index2Node


def __DensePageRank(nodeDictionary, deadNodeSet, args):
    # Generate the Matrix M
    node2Index, index2Node = __NodeIndexMatch(nodeDictionary, deadNodeSet)
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
    iterationTimes = 0
    nodeWeightPast = numpy.average(matrixMK, axis=1)
    iterationChanges = []

    while iterationTimes < args.max_iteration_times:
        if args.power_flag:
            iterationTimes *= 2
            matrixMK = numpy.matmul(matrixMK, matrixMK)
        else:
            iterationTimes += 1
            matrixMK = numpy.matmul(matrixMK, matrixM)
        nodeWeight = numpy.average(matrixMK, axis=1)

        iterationChanges.append([iterationTimes, numpy.sum(numpy.abs(nodeWeight - nodeWeightPast))])
        print('Iteration %d : Change =%.20f' % (iterationTimes, iterationChanges[-1][1]))
        if numpy.sum(numpy.abs(nodeWeight - nodeWeightPast)) < args.min_changes:
            break
        else:
            nodeWeightPast = nodeWeight
    return iterationChanges, nodeWeight, index2Node


def __SparsePageRank(nodeDictionary, deadNodeSet, args):
    node2Index, index2Node = __NodeIndexMatch(nodeDictionary, deadNodeSet)
    nodeWeightPast = numpy.ones(len(node2Index.keys())) / len(node2Index.keys())
    iterationChanges = []

    for iterationTimes in range(args.max_iteration_times):
        nodeWeight = numpy.zeros(len(node2Index.keys()))
        for linkStart in nodeDictionary.keys():
            if 0 < args.teleport_parameter <= 1:
                nodeWeight += nodeWeightPast[node2Index[linkStart]] * (1 - args.teleport_parameter) / len(node2Index.keys())
                for linkEnd in nodeDictionary[linkStart]:
                    nodeWeight[node2Index[linkEnd]] += nodeWeightPast[node2Index[linkStart]] / len(nodeDictionary[linkStart]) * args.teleport_parameter
            else:
                for linkEnd in nodeDictionary[linkStart]:
                    nodeWeight[node2Index[linkEnd]] += nodeWeightPast[node2Index[linkStart]] / len(nodeDictionary[linkStart])
        if not args.not_dead_end_flag:
            nodeWeight += (1 - numpy.sum(nodeWeight)) / len(node2Index.keys())
        iterationChanges.append([iterationTimes, numpy.sum(numpy.abs(nodeWeight - nodeWeightPast))])

        print('Iteration %d : Change =%.20f' % (iterationTimes, iterationChanges[-1][1]))
        if numpy.sum(numpy.abs(nodeWeight - nodeWeightPast)) < args.min_changes:
            break
        else:
            nodeWeightPast = nodeWeight

    return iterationChanges, nodeWeight, index2Node


def __SparseBlockPageRank(rawData, args):
    if not os.path.exists('Tmp'): os.makedirs('Tmp')
    reloadTimes = 0
    fileNumber = int(len(rawData) * 1.0 / args.block_length) + 1
    node2Index, index2Node, nodeOutDegree = {}, {}, {}

    file = open('Tmp/Tmp0.csv', 'w')
    for index in range(len(rawData)):
        if index % args.block_length == 0:
            file = file.close()
            file = open('Tmp/Tmp%d.csv' % int(index / args.block_length), 'w')
            reloadTimes += 1
        file.write(rawData[index])
        startNode, endNode = int(rawData[index].split('\t')[0]), int(rawData[index].split('\t')[1])
        if startNode not in node2Index.keys():
            node2Index[startNode] = len(node2Index.keys())
            index2Node[node2Index[startNode]] = startNode
        if endNode not in node2Index.keys():
            node2Index[endNode] = len(node2Index.keys())
            index2Node[node2Index[endNode]] = endNode
        if startNode not in nodeOutDegree.keys():
            nodeOutDegree[startNode] = 1
        else:
            nodeOutDegree[startNode] += 1
    file.close()

    nodeWeightPast = numpy.ones(len(node2Index.keys())) / len(node2Index.keys())
    iterationChanges = []

    for iterationTimes in range(args.max_iteration_times):
        if 0 < args.teleport_parameter < 1:
            nodeWeight = numpy.ones(len(node2Index.keys())) / len(node2Index.keys()) * args.teleport_parameter
        else:
            nodeWeight = numpy.zeros(len(node2Index.keys()))
        for batchIndex in range(fileNumber):
            data = numpy.genfromtxt(fname='Tmp/Tmp%d.csv' % batchIndex, dtype=int, delimiter='\t').reshape([-1, 2])
            if 0 < args.teleport_parameter < 1:
                for [startPoint, endPoint] in data:
                    nodeWeight[node2Index[endPoint]] += args.teleport_parameter * nodeWeightPast[node2Index[startPoint]] / nodeOutDegree[startPoint]
            else:
                for [startPoint, endPoint] in data:
                    nodeWeight[node2Index[endPoint]] += nodeWeightPast[node2Index[startPoint]] / nodeOutDegree[startPoint]

        if not args.not_dead_end_flag:
            nodeWeight += (1 - numpy.sum(nodeWeight)) / len(node2Index.keys())
        iterationChanges.append([iterationTimes, numpy.sum(numpy.abs(nodeWeight - nodeWeightPast))])

        print('Iteration %d : Change =%.20f' % (iterationTimes, iterationChanges[-1][1]))
        if numpy.sum(numpy.abs(nodeWeight - nodeWeightPast)) < args.min_changes:
            break
        else:
            nodeWeightPast = nodeWeight

    print('Disc Reload Times =', reloadTimes)
    return iterationChanges, nodeWeight, index2Node


if __name__ == '__main__':
    PageRankMain(
        '--input_path="D:/PythonFiles/WebBigData_PageRanke/WikiData.txt" --output_path="D:/PythonFiles/WebBigData_PageRanke" --teleport_parameter=0.85 --min_changes=1E-3 --max_iteration_times=1000 --top_node_number=100 --block_flag --block_length=1000')
