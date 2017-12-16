print("#################### 面向对象 ####################")
class Student(object):
    # 有__init__，在创建实例的时候，就不能传入空的参数了.注意：特殊方法“init”前后有两个下划线！！！
    def __init__(self, name, score, age):
        self.name = name
        self.score = score
        # 以__开头，就变成了一个私有变量（private）
        self.__age = age

    def get_age(self):
        return self.__age

    def set_age(self, age):
        self.__age = age

    def print_score(self):
        print('%s: %s - %s' % (self.name, self.score, self.__age))



bart = Student('Bart Simpson', 59, 20)
lisa = Student('Lisa Simpson', 87, 18)

bart
# <__main__.Student object at 0x10a67a590>
Student
# <class '__main__.Student'>


bart.print_score()

bart.score = 90
bart.print_score()


# bart.__age
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'Student' object has no attribute '__name'
bart.set_age(28)
print(bart.get_age())



print("#################### 多态 ####################")

class Animal(object):
    def run(self):
        print('Animal is running...')

# 继承了Animal
class Dog(Animal):
    # 子类的run()覆盖了父类的run()
    def run(self):
        print('Dog is running...')

class Cat(Animal):
    pass


dog = Dog()
dog.run()

cat = Cat()
cat.run()


# 对于静态语言（例如Java）来说，如果需要传入Animal类型，则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法。
# 对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了：



# 判断对象类型，使用type()函数：
type(123)
# <class 'int'>
type(dog)
# <class '__main__.Dog'>


# 判断class的类型，可以使用isinstance()函数。
isinstance(dog, Dog)
# True
isinstance('abc', str)
# True


# len()函数
len('ABC')
# 3
'ABC'.__len__()
# 3

