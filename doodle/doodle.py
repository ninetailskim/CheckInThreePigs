import math

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
        self.th = th
        self.r = r
        self.thabs = thabs
        self.th0 = th0
        self.parent = None
        self.children = []

    def getpos(self):
        return (self.x, self.y)

def toNodes(tree):
    nodes = []



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
        for node in nodes:
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
            y = anchor.nodeyx + math.sin(anchor.th+anchor.node.thabs) * anchor.r
            xw += x * anchor.w
            yw += y * anchor.w
        skin.x = xw
        skim.y = yw
