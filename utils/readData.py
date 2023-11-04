def getData():
    """
        data包含多个路径集合，每个路径集合是一个数据条目，表示两个实体间的路径集合
        data每一个路径集合包含多条路径，如下是一个例子(表示一对实体之间的多条路径)：
        [[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7], [1, 7, 8, 8, 3, 3, 4, 4, 5, 5, 6, 6, 7], [1, 9, 9, 10, 3, 3, 4, 4, 5, 5, 6, 6, 7], [1, 11, 10, 12, 3, 3, 4, 4, 5, 5, 6, 6, 7]]
        注意：
            1. 每个路径长度不定
            2. 从下标0开始计算，每个路径的偶数下标是实体id, 奇数下标是关系id
            3. 实体id和关系id都是从1开始的，所以实体和关系之间可能有重复，使用前一定要对关系重新标id，可以是加上实体id总数的偏置
        laebl是一个标签列表，表示每个路径集合的涉及到的实体对的可利用性关系成立概率。
        label[i]是一个可以利用性概率值，是data[i]涉及到的实体对之间的那个漏洞的可利用性
    """
    data = []
    label = []
    with open("../data/data.txt", "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.split()
            epss = float(line[0])
            label.append(epss)
            tpPathList = line[1].split("|")
            dataPath = []
            for pathStr in tpPathList:
                path = list(map(int, pathStr.split(",")))
                dataPath.append(path)
            data.append(dataPath)
    return data, label

data, label = getData()
