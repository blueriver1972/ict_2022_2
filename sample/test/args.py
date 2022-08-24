class Parent:
    def __init__(self, *args, **kwargs):
        print("Parent __init__")
        
    def imp(self):
        print("imp() in Parent")
        
        
class Child(Parent):
    def __init__(self, *args, **kwargs):
        super().__init__()            # => Parent().__init__()
        print(args)
        n1, n2, n3 = args
        print(f"{n1}, {n2}, {n3}")
        print(kwargs)
        print(kwargs["name"])
    
if __name__ =="__main__":
    
    
    child = Child(1, 2, 3, name="str", que="")
    child.imp()
    
    print("="*10)
    child = Child(3, 4, 5, que_1="", name= "str" )
    