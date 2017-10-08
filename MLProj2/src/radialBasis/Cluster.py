class Cluster:
    mean = []
    cluster_points = [[]]
    hasChanged = True
    def __init__(self, mean):
        self.mean = mean
    def addPoint(self, point):
        self.cluster_points.append(point)
    def calcCentroid(self):
        count = 0
        sum = [len(self.cluster_points)]
        for i in range(len(self.cluster_points)):
            for j in range(len(self.cluster_points[i])):
                sum[i] += self.cluster_points[i][j]
        for k in sum:
            sum[k] /= len(self.cluster_points)
        if self.mean == sum:
            hasChanged = False
        self.mean = sum
        self.cluster_points.clear()