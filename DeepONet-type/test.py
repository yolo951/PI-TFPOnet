

# 定义一些简单的函数
def total(a):
    def func(x):
        return x*a
    return func

# 创建一个包含函数的列表
mylist = [total(1), total(2), total(3)]

# 通过索引访问并调用列表中的函数
result1 = mylist[0](5)   # 相当于调用 func1(5)
result2 = mylist[1](5)   # 相当于调用 func2(5)
result3 = mylist[2](5)   # 相当于调用 func3(5)

print(result1)  # 输出 6
print(result2)  # 输出 10
print(result3)  # 输出 25
