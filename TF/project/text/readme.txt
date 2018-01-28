数据初始化

    读取文件


读取数据load_data_and_labels（地址）
    读取文件，UTF-8
    根据回车分组

    前后去空格
    拼接
    去除特殊符号clean_str（）

    创建标签
    拼接标签，concatenate

数据切分
    读取数据load_data_and_labels

    取得最大长度单词，并把其他补零

    数据洗牌，避免重复

    分割数据xy


网络架构TextCNN
            sequence_length：最大单词数量 56
            num_classes：分组数 2
            vocab_size：词汇数 18758
            embedding_size： 128
            filter_sizes： [3,4,5]
            num_filters： 128
            l2_reg_lambda：0.0

    输入、分类、dropout、loss





创建Session
    config配置

    Embedding

