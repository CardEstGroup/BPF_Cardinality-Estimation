# 查询语句的解析
# 给定一个输入的查询语句
# imdb：SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1990;
# stats：SELECT COUNT(*) FROM votes as v, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = v.UserId AND v.BountyAmount<=100 AND c.CreationDate>='2010-10-01 20:45:26'::timestamp AND c.CreationDate<='2014-09-05 12:51:17'::timestamp;
# 将其转化为对应属性的可能取值对应的标签？
# 得先确定每一种可能取值对应的标签
# 得先确定单表基数估计的方法？
# 将输入的SQL查询语句，转换为：
# 1. 表名对应表缩写的字典
# 2. k个连接条件（左表名缩写、左表连接属性、连接类型、右表名缩写、右表连接属性）
# 3. 查询条件（表名缩写、属性、查询条件）
# 4. 一个支持or析取操作的算法（and优先级比or高，一般来讲需要加上括号进行进一步的处理）
# 先实现1-3
import sqlparse
from sqlparse.tokens import Keyword,Name,Comparison
from sqlparse.sql import Where,Identifier

class Attr():
    def __init__(self,table,attr):
        self.table=table
        self.attr=attr
    def __hash__(self) -> int:
        return (self.table+self.attr).__hash__()
    def __eq__(self, value: object) -> bool:
        return ((self.table==value.table) and (self.attr==value.attr))
    def __repr__(self) -> str:
        return f'Attr({self.table}.{self.attr})'

# 先实现功能，再封装类或者函数
def parse_sql(sql,tolower=False):
    # sql=sql.replace('::timestamp','')
    stmt=sqlparse.parse(sql)[0] # 如需批量处理可以写成for stmt in stmts循环
    table_alias=dict()
    select_conditions=[]
    join_conditions=[]
    # 循环遍历处理每一个token
    after_from=False
    for token in stmt.tokens:
        if(token.match(Keyword,'FROM')):
            after_from=True
        if(not after_from):
            continue # 跳过到FROM关键词之后
        if(token.is_group): # 还有子token
            if(isinstance(token,Where)):
                # WHERE子句中识别连接条件和查询条件
                after_join=False
                for t in token.tokens:
                    if(t.is_keyword and t.value!='WHERE'): # 识别AND、OR、LIKE
                        if(after_join):
                            select_conditions.append(t.value)
                        else:
                            join_conditions.append(t.value)
                    if(t.is_group): # 识别连接条件和查询条件
                        a_condition=[]
                        for n in t.tokens: 
                            if(n.is_group):
                                names=[]
                                for an in n.tokens:
                                    if(an.ttype==Name):
                                        names.append(an.value)
                                if(len(names)==2):
                                    a_condition.append(Attr(names[0],names[1]))
                            if n.ttype==Comparison:
                                a_condition.append(n.value)
                        if(len(a_condition)==2):
                            after_join=True
                            a_condition.append(t.tokens[-1].value)
                            select_conditions.append(a_condition)
                        else:
                            join_conditions.append(a_condition)
            else:
                # FROM操作之后WHERE子句之前，识别缩写和涉及的表名
                for t in token.tokens:
                    if(t.is_group):
                        names=[]
                        for n in t.tokens:
                            if(n.ttype==Name):
                                names.append(n.value)
                            if(n.is_group):
                                for an in n.tokens:
                                    if(an.ttype==Name):
                                        names.append(n.value)
                        if(len(names)==1):
                            names.append(names[0])
                        table_alias[names[1]]=names[0]
                    elif(t.ttype==Name):
                        for at in token.tokens:
                            if isinstance(at,Identifier):
                                table_alias[at.value]=t.value
                        break
    if tolower:
        for sc in select_conditions:
            if isinstance(sc,str):
                continue
            sc[0].attr=sc[0].attr.lower()   
        for jc in join_conditions:
            if isinstance(jc,str):
                continue
            jc[0].attr=jc[0].attr.lower()      
            jc[2].attr=jc[2].attr.lower()         
    return table_alias,select_conditions,join_conditions
if __name__=="__main__":
    sql1="SELECT COUNT(*) FROM movie_info mi,title t WHERE t.id=mi.movie_id AND t.production_year>1990;"
    sql2="SELECT COUNT(*) FROM votes as v, comments as c, users as u WHERE u.Id = c.UserId AND u.Id = v.UserId AND v.BountyAmount<=100 AND c.CreationDate>='2010-10-01 20:45:26'::timestamp AND c.CreationDate<='2014-09-05 12:51:17'::timestamp;"
    sql=sql1
    table_alias,select_conditions,join_conditions=parse_sql(sql)
    print(sql)
    print(table_alias)
    print(select_conditions)
    print(join_conditions)
    print('AvA!')
