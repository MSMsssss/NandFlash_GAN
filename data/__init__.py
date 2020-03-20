result = np.load(data_root_path + file_prefix + "_" + file_list[0] + ".npy").astype(np.float32)
print("%s load success" % file_list[0])
file_list.remove(file_list[0])

for file in file_list:
    result = np.concatenate((result,
                             np.load(data_root_path + file_prefix + "_" + file + ".npy").astype(np.float32)),
                            axis=0)
    print("%s load success" % file)