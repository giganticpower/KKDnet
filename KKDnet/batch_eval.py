from test import test_checkpoint
import os

checkpoint_path = './checkpoints/kkd_r18_ctw/'

checkpoint_path_list = os.listdir(checkpoint_path)
for checkepoint in checkpoint_path_list:
    try:
        checkepoint_input = os.path.join(checkpoint_path, checkepoint)
        test_checkpoint(checkpoint=checkepoint_input)
    except ZeroDivisionError as e:
        print("except:", e)
    finally:
        print(checkepoint)
print("over")