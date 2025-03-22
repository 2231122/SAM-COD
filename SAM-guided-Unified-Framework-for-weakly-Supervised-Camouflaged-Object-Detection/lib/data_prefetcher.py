import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target, _, _,self.x_h,self.y_w,self.next_target_gt  = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            device = torch.device("cuda:3")
            self.next_input = self.next_input.to(device)
            self.next_target = self.next_target.to(device)
            self.next_target_gt = self.next_target_gt.to(device)

            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_target_gt = self.next_target_gt.float()  # if need

            self.next_x_h=self.x_h.to(device)
            self.next_y_w=self.y_w.to(device)



    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        target_gt = self.next_target_gt

        x_h =self.next_x_h
        y_w=self.next_y_w
        self.preload()
        return input, target,x_h,y_w,target_gt
