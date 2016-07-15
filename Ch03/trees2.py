



if __name__ == '__main__':
    fr = open('trainData.csv')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # print(lenses)
lensesLabels1 = ['Demand', 'OrderQu', 'FillRate', 'Sale', 'Salepre', 'Averagestock', 'Stock_salesRatio']
lensesTree = trees.createTree(lenses, lensesLabels1)
print(lensesTree)
treePlotter.createPlot(lensesTree)