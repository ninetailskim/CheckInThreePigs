import math
import copy

FatherAndSon = {
    'thorax':'centerpoint',
    'upper_neck':'thorax',
    'head_top':'upper_neck',

    'left_shoulder':'thorax',
    'left_elbow':'left_shoulder',
    'left:wrist':'left_elbow',

    'right_shoulder':'thorax',
    'right_elbow':'right_shoulder',
    'right_wrist':'right_elbow',

    'pelvis':'centerpoint',
    'left_hip':'pelvis',
    'left_knee':'left_hip', 
    'left_ankle':'left_knee',

    'right_hip':'pelvis',
    'right_knee':'right_hip',
    'right_ankle':'right_knee',

    'centerpoint':'centerpoint'
}

def complexres(res):
    cres = copy.deepcopy(res)
    for key,pos in res.items():
        father = FatherAndSon[key]
        if key[0] == 'm' or father[0] == 'm':
            midkey = 'm'+key+'_'+father
        else:
            kn = ''
            for t in key.split('_'):
                kn += t[0]
            fn = ''
            for t in father.split('_'):
                fn += t[0]
            midkey = 'm_'+kn+'_'+'fn'
        midvalue = ((pos[0] + res[father][0]) / 2, (pos[1] + res[father][1])/2)
        FatherAndSon[key] = midkey
        FatherAndSon[midkey] = father
        cres[midkey] = midvalue


def distance(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))


def dist2weight(md):
    if len(md) == 1:
        return [1]
    maxx = max(md)
    minx = min(md) 
    s = 0
    f = []
    for imd in md:
        x = 0.1 + 0.9 * (imd - minx) / (maxx - minx)
        s += x
        f.append(x)
    
    return [x/s for x in f]


class skinItem():
    def __init__(self, x, y):
        super(skinItem, self).__init__()
        self.x = x
        self.y = y
        self.anchor = []

    def getpos(self):
        return (self.x, self.y)

    def appendAnchor(self, anchor):
        self.anchor.append(anchor)

    def getAnchor(self):
        return self.anchor


class anchorItem():
    def __init__(self, node, th, r, w):
        super(anchorItem, self).__init__()
        self.node = node
        self.th = th
        self.r = r
        self.w = w


class nodeItem():
    def __init__(self, x, y, th, r, thabs, th0):
        super(nodeItem, self).__init__()
        self.x = x
        self.y = y
        
        self.parent = None
        self.children = []

    def getpos(self):
        return (self.x, self.y)

    def setParent(self, parent):
        self.parent = parent

    def appendChildren(self, child):
        self.children.append(child)

    def setInfo(self, th, r, thabs, th0):
        self.th = th
        self.r = r
        self.thabs = thabs
        self.th0 = th0


def getScale():
    pass

def addcenterPoint(res):
    thorax = res['thorax']
    pelvis = res['pelvis']
    x = (thorax[0] + pelvis[0]) / 2
    y = (thorax[1] + pelvis[1]) / 2
    res['centerpoint'] = (x,y)
    return res



nodes = {}
def toNodes(tree):
    for key in FatherAndSon:
        nodes[key] = nodeItem(tree[key][0], tree[key][1])


def connectNodes(tree):
    for key,node in nodes.items():
        if key == 'centerpoint':
            continue
        if node.parent is not None:
            continue
        node.setParent(nodes[FatherAndSon[key]])
        nodes[FatherAndSon[key]].appendChildren(node)


def setInfo(node):
    if node.parent is None:
        node.setInfo(0,0,0,0)
    else:
        #和父节点之间的角度
        th = math.atan2(node.y-node.parent.y, node.x-node.parent.x)
        #和父节点之间的距离
        r = distance(node.parent.getpos(), node.getpos())
        #用和父节点之间的角度 剪掉 父节点的thabs
        #所以这个node.th是计算相对于父节点的相对角度,这个
        node.th = th - node.parent.thabs
        #和父节点的距离
        node.r = r
        #和父节点的相对角度
        node.thabs = th
        #th0应该被认为是th的初始值,后来就没有再改变过
        node.th0 = node.th 
    
    for n in node.children:
        setInfo(n)

#这里需要一个函数,拿到最新的骨骼图,
#从根节点去更新所有骨头的相对角度,那TMD不就是setInfo吗?好像问题就这么解决了



def buildskin(lines, nodes):
    if lines is None or nodes is None or len(lines) == 0 or len(nodes) == 0:
        return []
    skins = []
    for line in lines:
        for p in line:
            skins.append(skinItem(p[0], p[1]))
    
    for skin in skins:
        md = [float("inf"), float("inf"), float("inf"), float("inf")]
        mn = [None, None, None, None]
        mdlen = 0
        for key,node in nodes.items():
            d = distance(skin.getpos(), node.getpos())
            for imd in md:
                if d < imd:
                    mdlen += 1
                    md = [d] + md[:-1]
                    mn = [node] + mn[:-1]
                    break
        # skin.setAnchors 
        if mdlen < 4:
            md = md[:mdlen]
            mn = mn[:mdlen]
        ws = dist2weight(md)
        
        for j in range(len(mn)):
            th = math.atan2(skin.y-mn[j].y, skin.x-mn[j].x)
            r = distance(skin, mn[j])
            w = ws[j]
            skin.appendAnchor(anchorItem(mn[j], th-mn[j].thabs, r, w))

    return skins


def calculateSkin(skins):
    for skin in skins:
        xw = 0
        yw = 0
        for anchor in skin.getAnchor():
            x = anchor.node.x + math.cos(anchor.th+anchor.node.thabs) * anchor.r
            y = anchor.node.y + math.sin(anchor.th+anchor.node.thabs) * anchor.r
            xw += x * anchor.w
            yw += y * anchor.w
        skin.x = xw
        skim.y = yw


# 从这里看出来每次更新的时候其实要用到的,是anchor中的node的x,y,以及这个node的thabs
# 其实这里可以直接× scale

# anchor的xy肯定是要随着视频变化的,所以好像也不用去× scale???
# 错,还是需要×的,不然这个人会变得非常的细瘦, 而不是符合轮廓

# 这个scale要如何计算呢?一个是一开始的模板的长度,一个是现在视频里的长度,然后 anchor.r / templength * videolength
