class Cluster:
    mean = []
    hasChanged = True
    def __init__(self, mean):
        self.mean = mean
        self.clusterPoints = [[]]
    def addPoint(self, point):
        self.clusterPoints.append(point)
    def calcCentroid(self):
        count = 0
        sum = [0 for x in range(len(self.clusterPoints[0]))]
        print(len(sum))
        print(len(self.clusterPoints))
        print(len(self.clusterPoints[0]))
        for i in range(len(self.clusterPoints)):
            for j in range(len(self.clusterPoints[i]) - 1):
                temp = self.clusterPoints[i]
                sum[i] += temp[j]
        print(sum)
        for k in sum:
            k /= float(len(self.clusterPoints))
        if self.mean == sum:
            hasChanged = False
        self.mean = sum
        self.clusterPoints = [[]]
    def getMean(self):
        return self.mean