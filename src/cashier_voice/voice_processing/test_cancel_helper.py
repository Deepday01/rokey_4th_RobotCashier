#!/usr/bin/env python3

from robot_control.robot_cancel_helper import remove_item_by_pose

def main():
    #target_pose = [300.0, 100.0, 200.0, 156.4, 180.0, -112.5]   # 예시
    for i in range(5):
        if i == 0:            
            print(f"Test {i+1}")
            target_pose = [250.6, 277.1, 280.0, 90, 180, 45.5]
            success = remove_item_by_pose(target_pose)
            if success:
                print("remove_item_by_pose success")
            else:
                print("remove_item_by_pose failed")
        elif i == 1:   
            print(f"Test {i+1}")
            target_pose = [431.6, 140.7, 280.0, 90, 180, 30.1]
            success = remove_item_by_pose(target_pose)
            if success:
                print("remove_item_by_pose success")
            else:
                print("remove_item_by_pose failed")
        elif i == 2:
            print(f"Test {i+1}")
            target_pose = [287.4, 53.2, 280.0, 90, 180, 5.1]
            success = remove_item_by_pose(target_pose)
            if success:
                print("remove_item_by_pose success")
            else:
                print("remove_item_by_pose failed")
        elif i == 3:
            print(f"Test {i+1}")
            target_pose = [158.9, 152.5, 280.0, 90, 180, 70]
            success = remove_item_by_pose(target_pose)
            if success:
                print("remove_item_by_pose success")
            else:
                print("remove_item_by_pose failed")
    #drop_pose   = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]  # 예시
    #success = remove_item_by_pose(target_pose, drop_pose)
    # if success:
    #     print("remove_item_by_pose success")
    # else:
    #     print("remove_item_by_pose failed")

if __name__ == "__main__":
    main()