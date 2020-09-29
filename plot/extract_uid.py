import os
from os import path

filename = 'selected_data.txt'
selected_uid = []
raw_dataset = '/home/local/ASUAD/ychen404/Code/DeepMove_new/serm-data/tweets-cikm.txt'
save_output = 'tweets-cikm-50.txt'
done_uid = 0
def extract_data(file_raw, file_dst, target_uid):
    pid_cnt = []
    line_cnt = 0
    # with open(file_raw) as fid:
    #     with open(file_dst) as f_uid1:
    for i, line in enumerate(file_raw):
        # print(line)
        _, uid, _, _, tim, _, _, tweet, pid = line.strip('\r\n').split('')
        if uid == str(target_uid):
            # print("Found")
            file_dst.write(line)
            line_cnt += 1
            if pid not in pid_cnt:
                pid_cnt.append(pid)
            # f_log.write(str("uid_1" + '\t' + str(i) + '\t' + str(uid)) + '\t' + str(pid) + '\n')
        else:
            # print("Do notthing")
            pass
    return line_cnt, pid_cnt


with open(filename, "r") as f:
    for line in f:
        line = line.split('\t')
        print(line[1])
        picklename = line[1].split('.')[0]
        print(picklename)
        uid = picklename.split('-')[3]
        print(uid)
        selected_uid.append(uid)

print("The length of selected_uid is {}".format(len(selected_uid)))

if path.exists(save_output):
    print(f"{save_output} already exist, deleting..")
    os.remove(save_output)
    print(f"{save_output} deleted")

# for testing
# test_uid = ['16958072', '6338132']

# for uid in test_uid: # for testing
for uid in selected_uid:
    with open(raw_dataset, 'r') as fd_raw:
        with open(save_output, 'a+') as fd_output:
            _, _ = extract_data(fd_raw, fd_output, uid)
        print(f"UID {uid} done processing")
        done_uid +=  1
print(f"Processed a total of {done_uid} UID's")


# fd_raw.close()
# fd_output.close()