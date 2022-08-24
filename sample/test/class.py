
 

class Calculator:
    def __init__(self, name="calc"):
        self.name = name
        
    def __call__(self, text=""):
        print(f" a {self.name} {text}")
        
if __name__ == "__main__":      
    #pass

    calc1 = Calculator(name="calc1_callable")
    calc2 = Calculator(name="clac2=callable")()
    
    print(calc1())
    #print(calc2("ict_cog"))
    
    
 
