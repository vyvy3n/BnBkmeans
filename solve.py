"""
the class realizes a simple linear solver with the help of Gurobi
the input model can be a maximum or minmum problem
and the solver will return the result and optimal
"""
from gurobipy import *


class Solve(object):
    def __init__(self, number, constraints, obj, constant=0):
        """
        :param number: number means the number of variables
        :param constraints: all the constraints presented by a list of lists
        :param obj: the object function presented by a list
        :param constant: the additional constant in objective function, the default value is zero
        
        example: max 5x + 6y + 7
        x + y <= 7
        9x + 7y <= 9
        
        number is 2
        constraints is [[1, 1, '<=', 7], [9, 7, '<=', 9]]
        obj is ['max', 5, 6]
        constant is 7
        """
        self.number = number
        self.constraints = constraints
        self.obj = obj
        self.constant = constant
        self.m = Model('LP')
        self.m.setParam('OutputFlag', 0)

    def add_cons(self, l):
        """
        add constraints into the model
        
        :param l: the constraint list
        
        :return: nothing
        """
        linear_exp = self.generate_expression(l)
        if l[self.number] == '<=':
            tmp = GRB.LESS_EQUAL
        elif l[self.number] == '>=':
            tmp = GRB.GREATER_EQUAL
        elif l[self.number] == '=':
            tmp = GRB.EQUAL
        else:
            print('input invalid!!!')
            return None
        self.m.addConstr(linear_exp, tmp, l[self.number + 1])
        self.m.update()

    def add_obj(self, l):
        """
        add objective function into the model
        
        :param l: the objective function list
        
        :return: nothing
        """
        linear_exp = self.generate_expression(l[1:])
        linear_exp += self.constant
        if l[0] == 'max':
            tmp = GRB.MAXIMIZE
        elif l[0] == 'min':
            tmp = GRB.MINIMIZE
        else:
            print('input invalid!!!')
            return None
        self.m.setObjective(linear_exp, tmp)
        self.m.update()

    def generate_expression(self, l):
        """
        generating the expression by the variables of the model and the input list
        then adding this linear expression into the model
        
        :param l: l is the list that will be transformed
        
        :return: the linear expression
        """
        variables = self.m.getVars()
        linear_exp = LinExpr()
        for i in range(self.number):
            linear_exp += variables[i] * l[i]
        return linear_exp

    def add_variables(self):
        """
        add variables into the model
        :return: nothing
        """
        for i in range(self.number):
            self.m.addVar(vtype=GRB.CONTINUOUS, name='x' + str(i))
        self.m.update()

    def solve(self):
        """
        solve the linear program
        
        :return: the solution, opt
        """
        self.add_variables()
        self.add_obj(self.obj)
        for i in self.constraints:
            self.add_cons(i)
        self.m.optimize()
        if self.m.status == GRB.OPTIMAL:
            x = self.m.getAttr('x')
            opt = self.m.objVal
            return x, opt
        elif self.m.status == GRB.INFEASIBLE:
            return 'infeasible', 0
        elif self.m.status == GRB.UNBOUNDED:
            return 'unbounded', 0
        elif self.m.status == GRB.INF_OR_UNBD:
            return 'infeasible', 'unbounded'
