from LoadCfg import NetManager
from tensorboardX import SummaryWriter


class Mytensorboard(NetManager):
    def __init__(self):
        self.writer = SummaryWriter()
        self.writerLayout = { 'Loss': {} }
        self.step = 0

    @staticmethod
    def setObjectStep(self, num_set):
        self.object_step = num_set * self.OBJECT_EPOCH

    def plotScalars(self):
        for key, values in self.writerLayout.items():
            self.writer.add_scalars(key, values, self.step)

    def SetLoss(self, name, value):
        self.writerLayout['Loss'][name] = value