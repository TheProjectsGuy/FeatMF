"""
    An example sub-module for documentation purposes
"""

def say_hello(user: str="World"):
    """
        Say hello to a user.
        
        :param user:   The user to say hello to.
    """
    print(f"Hello, {user}!")


class Person:
    """
        A class that represents a person.
    """
    def __init__(self, name: str="World") -> None:
        """
            Initialize the person.
            
            :param name:    The name of the person.
        """
        self.name = name
    
    def get_name(self) -> str:
        """
            Get the name of the person.
        """
        return self.name

class HelloTeller:
    """
        A class that greets people.
    """
    def __init__(self, name: Person=Person(), count: int=2) -> None:
        """
            Initialize the greeter.
            
            :param name:    The person to greet. Default: ``Person()``.
            :type name:     Person
            :param count:   The number of times to greet the person.
        """
        self.name = name
        self.count = count
    
    def greet(self) -> None:
        """
            Say hello to the person. Very similar to :py:func:`say_hello`.
        """
        for _ in range(self.count):
            print(f"Hello, {self.name}!")
    
    def get_person(self) -> Person:
        """
            Get the name of the person.
            
            :return:    The person.
        """
        return self.name
