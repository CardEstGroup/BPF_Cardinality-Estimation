class AttributePair:
    def __init__(self,key,value):
        self.key=key
        self.value=value
    def __repr__(self):
        return f'({self.key},{self.value})'
    def __hash__(self):
        l=[self.key,self.value]
        l.sort()
        return tuple(l).__hash__()
    def get_another(self, key):
        if key==self.key:
            return self.value
        elif key==self.value:
            return key