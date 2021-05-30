
def apply_minimum_thresh_hold(Data, Candinatek, minSupport):
    ssCnt = {}
    for tid in Data:
        for can in Candinatek:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(Data))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


def create_candinate_set(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return list(map(frozenset, C1))  # use frozen set so we
    # can use it as a key in a dict



def aprioriGenration(frequent_item_set, k): #creates Ck
    retList = []
    lenLk = len(frequent_item_set)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(frequent_item_set[i])[:k-2]; L2 = list(frequent_item_set[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(frequent_item_set[i] | frequent_item_set[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    Candinate = create_candinate_set(dataSet)
    Data = list(map(set, dataSet))
    frequent_set, supportData = apply_minimum_thresh_hold(Data, Candinate, minSupport)
    frequent_set = [frequent_set]
    combination_length = 2
    while (len(frequent_set[combination_length-2]) > 0):
        Candinate_set = aprioriGenration(frequent_set[combination_length-2], combination_length)
        frequent_setk, supK = apply_minimum_thresh_hold(Data, Candinate_set, minSupport)#scan DB to get Lk
        supportData.update(supK)
        frequent_set.append(frequent_setk)
        combination_length += 1
    return frequent_set, supportData

def calculate_confidence(frequent_set, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[frequent_set]/supportData[frequent_set-conseq] #calc confidence
        if conf >= minConf:
            print (frequent_set-conseq,'-->',conseq,'conf:',conf)
            brl.append((frequent_set-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rules_fromation(frequent_set, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(frequent_set) > (m + 1)): #try further merging
        Hmp1 = aprioriGenration(H, m+1)
        Hmp1 = calculate_confidence(frequent_set, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rules_fromation(frequent_set, Hmp1, supportData, brl, minConf)

def generateRules(frequent_set, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(frequent_set)):
        for frequent_set in frequent_set[i]:
            H1 = [frozenset([item]) for item in frequent_set]
            if (i > 1):
                rules_fromation(frequent_set, H1, supportData, bigRuleList, minConf)
            else:
                calculate_confidence(frequent_set, H1, supportData, bigRuleList, minConf)
    return bigRuleList