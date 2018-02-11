import collections
import re

"""
求解：argmaxc P(c|w) -> argmaxc P(w|c) P(c) / P(w)
P(c), 文章中出现一个正确拼写词 c 的概率, 也就是说, 在英语文章中, c 出现的概率有多大
P(w|c), 在用户想键入 c 的情况下敲成 w 的概率. 因为这个是代表用户会以多大的概率把 c 敲错成 w
argmaxc, 用来枚举所有可能的 c 并且选取概率最大的
"""


# 把语料中的单词全部抽取出来, 转成小写, 并且去除单词中间的特殊符号
def words(text):
    return re.findall('[a-z]+', text.lower())


# 统计词频
def train(features):
    # 避免碰到没有出现过的词，而概率产生0的情况。默认所有出现次数为1，如果更多就+1
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model


# {'the': 80031, 'project': 289, ...}
NWORDS = train(words(open('big.txt').read()))

"""
要是遇到我们从来没有过见过的新词怎么办. 假如说一个词拼写完全正确, 但是语料库中没有包含这个词, 
从而这个词也永远不会出现在训练集中. 于是, 我们就要返回出现这个词的概率是0. 这个情况不太妙, 
因为概率为0这个代表了这个事件绝对不可能发生, 而在我们的概率模型中, 我们期望用一个很小的概率来代表这种情况. lambda: 1
"""

alphabet = 'abcdefghijklmnopqrstuvwxyz'

"""
根据编辑举例判断用户可能出现的错误

编辑距离:
两个词之间的编辑距离定义为使用了几次
插入(在词中插入一个单字母), 删除(删除一个单字母), 交换(交换相邻两个字母), 替换(把一个字母换成另一个)的操作从一个词变到另一个词.
"""
# 返回所有与单词 w 编辑距离为 1 的集合
def edits1(word):
    n = len(word)
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion 删除
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition 交换
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration 插入一个字母
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion 替换一个字母


"""
与 something 编辑距离为2的单词居然达到了 114,324 个
优化:在这些编辑距离小于2的词中间, 只把那些正确的词作为候选词,只能返回 3 个单词: ‘smoothing’, ‘something’ 和 ‘soothing’
"""
# 返回所有与单词 w 编辑距离为 2 的集合
# 在这些编辑距离小于2的词中间, 只把那些正确的词作为候选词
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

"""
正常来说把一个元音拼成另一个的概率要大于辅音 (因为人常常把 hello 打成 hallo 这样); 把单词的第一个字母拼错的概率会相对小, 等等.
但是为了简单起见, 选择了一个简单的方法: 编辑距离为1的正确单词比编辑距离为2的优先级高, 而编辑距离为0的正确单词优先级比编辑距离为1的高.
"""
def known(words): return set(w for w in words if w in NWORDS)


# 根据输入单词，计算最可能的输入。
# 如果known(set)非空, candidate 就会选取这个集合, 而不继续计算后面的
def correct(word):
    # 判断最可能编辑距离为1、2的单词，和本身。 并获得词频
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    # 返回词频最大的一个
    return max(candidates, key=lambda w: NWORDS[w])


print(correct('appl'))
print(correct('appla'))
print(correct('learw'))
print(correct('tess'))
print(correct('morw'))
