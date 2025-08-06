class Animal():
    def __init__(self, name):
        print("Animal created")
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement this abstract method")


class Bird(Animal):
    def __init__(self, name):
        print("Bird created")
        super().__init__(name)

    def speak(self):
        print(f"{self.name} says chirp!")

class Mamal(Animal):
    def __init__(self, name):
        print("Mamal created")
        super().__init__(name)

    def speak(self):
        print(f"{self.name} says hello!")


class Amphibian(Animal):
    def __init__(self, name):
        print("Amphibian created")
        super().__init__(name)

    def speak(self):
        print(f"{self.name} says hello!")


class Domestic():
    def __init__(self, name):
        print("Domestic created")
        self.name = name

    def intro(self):
        print(f"I, {self.name}, am a domestic animal.")

    def speak(self):
        print(f"Domestic {self.name} says hello!")


class Wild():
    def __init__(self, name):
        print("Wild created")
        self.name = name

    def intro(self):
        print(f"Beware!!! I, {self.name}, am a wild animal.")


class Dog(Mamal, Domestic):
    def __init__(self, name):
        print("Dog created")
        super().__init__(name)
        Domestic.__init__(self, name)

    def intro(self):
        print(f"{self.name} woofs")

class Cat(Mamal, Domestic):
    def __init__(self, name):
        print("Cat created")
        super().__init__(name)
        Domestic.__init__(self, name)

    def intro(self):
        print(f"{self.name} meows")


if __name__ == "__main__":
    mydog = Dog("Tommy")
    mydog.speak()
    mydog.intro()

    mycat = Cat("Kitty")
    mycat.speak()
    mycat.intro()

    # MRO method returns the method resolution order
    """MRO determines the sequence in which base classes are searched 
    when a method or attribute is accessed on an object. 
    This is crucial in multiple inheritance scenarios 
    where a method might be defined in more than one parent class.
    """
    print(f"Cat.mro(): {Cat.mro()}")

    Domestic.speak(mycat)