# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:34:59 2021

@author: Dylan

This code is intended to assign students with similar research interests to
Zoom break-out rooms, so they can meet. If two students are from the same
institute, they probably already know each other, and putting them in the same
break-out room should be penalised. The meeting host is assigned to 'room 0', 
the main Zoom-room: this is so participants always know where to find the host. 
The code was built for a student matching event, but should work for any Zoom
meeting in which participants should be divided into break-out rooms according
to interests. The output is a break-out room per person, along with the top
three topics for that room.

USAGE:
1. Tell participants to fill out their details in an online spreadsheet,
    formatted like 'anonymous example.csv'. You can propose a list of topics to
    choose from, but this is not binding: the code will match strings that
    look similar.
2. Download that spreadsheet as a csv. Point the parameter 'standard_in' to
    that csv.
3. Configure the other parameters below, and run the script. If Gurobi is 
    installed, this may save some computing time.
4. Read out and share the output-file, probably named 'results.csv'.
    
PARAMETERS:
- same_institute_penalty: Set this to 0 if it is okay for people from the same
    group to meet, otherwise, leave it at 10 or another high number.
- min_per_room: Minimum number of people that should be assigned per room.
- max_per_room: Maximum number of people that should be assigned per room.
- nr_tags: How many topics/tags you have allowed participants to enter.
- csv_separator: If you open the csv's with a text editor, which separator 
    shows up.
- standard_in: Path to the input csv.
- standard_out: Path where you would like the output csv to show up.
- standard_matrix: Path where you would like the distance matrix to show up.
- nr_shown_themes: How many topics are presented when describing the room.
- tag_multiplier: For any two people, their best matching tags should hold more
    weight than the other tags. This factor indicates how many times as
    important the best match is than the second-best, and the second-best than
    the third, ...
- preassigned_room: If you want to prescribe certain people to certain rooms,
    this dict allows that.
    
MATH:
    
If person x and person y have entered tag lists t_x and t_y of length at most
d, the distance between D(x,y) between x and y is computed as follows.
- A matrix M is compiled of edit distances (between 0 and 1) between strings 
    in t_x and t_y.
- A minimum weight bipartite matching is made between t_x and t_y based on M.
- The distances between matched tags are weighted, summed and normalised. If 
    min_t is min( len(t_x), len(t_y) ), and t_x and t_y are sorted such that
    t_x(i) is matched to t_y(i) and 
    M(t_x(1), t_y(1)) <= M(t_x(2), t_y(2)) <= ...,
    then we compute
        (sum_(i=1)^(min_t) (tag_multiplier)^(min_t - 1) M(t_x(i), t_y(i)) ) 
        / (sum_(i=1)^(min_t) (tag_multiplier)^(min_t - 1))
    which is again between 0 and 1.
- This distance between x and y is increased by same_institute_penalty if x and
    y are in the same institute.
Note that, by construction, anyone has distance 0 to themselves.
    
Once the distance matrix between people is determined, the assignment of
participants to break-out rooms is computed with the following integer linear
program:
    
x(i,b) in {0,1}: whether student i is assigned to break-out room b
y(i,j) in {0,1}: whether students i and j are in the same break-out room
    
min sum_(i,j) D(i,j) * y(i,j)
s.t. 
sum_(i,b) x(i,b) = 1, (forall i) (1)
sum_(i,b) x(i,b) <= max_per_room, (forall b) (2)
sum_(i,b) x(i,b) >= min_per_room, (forall b) (3)
x(i,b) + x(j,b) <= y(i,j) + 1, (forall i, j, b) (4)
x(i,b) = 1, (forall fixed assignments (i,b) in preassigned_room)
x(i,b) in {0,1}, y(i,j) in {0,1}

Constraint (1) states that everyone should get exactly one room.
Constraint (2) states that a room can receive no more than max_per_room
    participants, and (3) states that it should get no less than min_per_room.
Constraint (4) activates y(i,j) when i and j are assigned to the same room.
Constraint (5) enacts the preassigned room assignments.

The variables y(i,j) might as well be continuous, but they're left binary for
clarity.

"""

# Configurable settings:

same_institute_penalty = 10
min_per_room = 3
max_per_room = 4
nr_tags = 3
csv_separator = ','
standard_in = 'exampleinput.csv'
standard_out = 'results.csv'
standard_matrix = 'similarities.csv'
nr_shown_themes = 3
tag_multiplier = 3
preassigned_room = {'Dylan Huizing_CWI': 'Break-out room 0'}

# Libraries:

from difflib import SequenceMatcher
from math import ceil
from pandas import DataFrame, read_csv
from pulp import lpSum, LpProblem, LpVariable, LpMinimize, LpBinary
from pulp import value as pulpvalue
from scipy.optimize import linear_sum_assignment
from traceback import print_exc

class student:
    
    def __init__(self, name=None, tags=None, institute=None):
        
        self.name = name
        self.tags = tags
        self.institute = institute
        
    def __str__(self):
        
        return str(self.__dict__)

class instance:
    
    def __init__(self, students=None):
        
        self.students = students
        
    def __getitem__(self, key):
        return self.students[key]
        
    def __len__(self):
        
        return len(self.students)
        
    def read(filename=standard_in):
        
        DF = read_csv(filename, sep=csv_separator)
        
        students = list()
        for i, name in enumerate(DF['NAME']):
            try:
                s = student()
                s.name = name
                s.institute = DF['INSTITUTE'][i]
                s.tags = list()
                for t in range(1, nr_tags+1):
                    tag = DF['TAG %d' % t][i]
                    if isinstance(tag, str) and len(tag) > 0:
                        if len(s.tags) == 0:
                            s.tags = [tag]
                        else:
                            s.tags += [tag]
                students += [s]
            except:
                print('Rejected student on line %d.' % i)
                print_exc()
                
        return instance(students)
        
def generate_breakoutrooms(I):
    
    m = ceil( len(I) / max_per_room )
    B = ['Break-out room %d' % b for b in range(m)]
    return B

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def distance(I):
    
    D = {i: {j: 0 for j in I} for i in I}
    
    for ind, i in enumerate(I):
        for j in I[ ind+1: ]:
            D[i][j] += tags_distance(i.tags, j.tags)
            if i.institute == j.institute:
                D[i][j] += same_institute_penalty
            D[j][i] = D[i][j]
            
    return D
    
def tags_distance(taglist1, taglist2):

    # Empty tag lists get fully bad score:    
    if len(taglist1) == 0 or len(taglist2) == 0:
        return 1
    
    # Match the tags optimally using the Hungarian algorithm:
    M = [[1 - similar(ti, tj) for tj in taglist2] for ti in taglist1]
    row_ind, col_ind = linear_sum_assignment(M)
    
    # Take a weighted average of the similarities, where the best matched
    # tag is most important and the worst matched is least:
    min_t = min(len(taglist1), len(taglist2))
    dist = 0
    dists = [ M[ row_ind[ind] ][ col_ind[ind] ] for ind in range(min_t)]
    sorted_dists = sorted(dists, reverse=True)
    multiplier = 1
    normaliser = 0
    for d in sorted_dists:
        dist += multiplier*d
        normaliser += multiplier
        multiplier *= tag_multiplier
    dist /= normaliser

    return dist
    
def optimal_assignment(I, B, D):
    
    S = ['%s_%s' % (i.name, i.institute) for i in I]
    Ds = {i: {j: D[ I[ind_i] ][ I[ind_j] ] for ind_j, j in enumerate(S)}
          for ind_i, i in enumerate(S)}
    
    mip = LpProblem('matchmaking', LpMinimize)
    
    mip.x = x = LpVariable.dicts('x', (S,B), cat=LpBinary)
    mip.y = y = LpVariable.dicts('y', (S,S), cat=LpBinary)
    
    # Objective function:
    mip += lpSum(Ds[i][j]*y[i][j] for i in S for j in S)
    
    # (1): everyone must be assigned exactly one room:
    for i in S:
        mip += lpSum(x[i][b] for b in B) == 1, "(1)_%s" % i
        
    # (2): each room has a maximum size:
    for b in B:
        mip += lpSum(x[i][b] for i in S) <= max_per_room, "(2)_%s" % b
        
    # (3): each room has a minimum size:
    for b in B:
        mip += lpSum(x[i][b] for i in S) >= min_per_room, "(3)_%s" % b
        
    # (4): if students are assigned the same room, they're in the same room:
    for i in S:
        for j in S:
            for b in B:
                mip += x[i][b] + x[j][b] <= y[i][j] + 1, "(4)_%s_%s_%s" % (i,j,b)
                
    # (5): some people have preassigned rooms:
    for i in preassigned_room:
        b = preassigned_room[i]
        if (not i in S) or (not b in B):
            print('Preassigned room ignored for unknown student %s.' % i)
        else:
            mip += x[i][b] == 1, "(5)_%s" % (i)
            
    # Solve with Gurobi if possible, otherwise whatever PuLP deems best:
    try:
        from pulp import GUROBI
        GUROBI().solve(mip)
    except:
        print('Gurobi solver not found.')
        mip.solve()
        
    # Read out results:
    x_vals = {i: {b: pulpvalue(x[ S[ind_i] ][b]) for b in B} 
              for ind_i, i in enumerate(I)}
    y_vals = {i: {j: pulpvalue(y[ S[ind_i] ][ S[ind_j] ]) 
                  for ind_j, j in enumerate(I)} for ind_i, i in enumerate(I)}
    
    return (x_vals, y_vals)
    
def room_descriptions(I, B, D, x, y):
    
    students_of_room = {b: [i for i in I if x[i][b]] for b in B}
    
    description_of_room = {b: '' for b in B}
    for b in B:
        
        all_themes = set()
        count = dict()
        for i in students_of_room[b]:
            for t in i.tags:
                all_themes.add(t)
                if not t in count:
                    count[t] = 0
                count[t] += 1
                
        popular_themes = sorted(all_themes, key = lambda t: count[t], 
                                reverse=True)
        chosen = popular_themes[ :nr_shown_themes ]
        
        for t in chosen[:-1]:
            description_of_room[b] += '%s, ' % t
        description_of_room[b] += chosen[-1]
        
    return description_of_room

def print_result(I, B, D, x, y, description_of_room):
    
    room_of_student = {i: next(b for b in B if x[i][b]) for i in I}
    
    sorted_I = sorted(I, key = lambda i: i.name)
    for i in sorted_I:
        b = room_of_student[i]
        themes = description_of_room[b]
        print('%s (%s): %s (%s)' % (i.name, i.institute, b, themes))
        
def save_match_matrix(I, D, outlocation=standard_matrix):
    
    S = ['%s_%s' % (i.name, i.institute) for i in I]
    Ds = {i: {j: D[ I[ind_i] ][ I[ind_j] ] for ind_j, j in enumerate(S)}
          for ind_i, i in enumerate(S)}
    
    DF = DataFrame(Ds)
    DF.to_csv(outlocation)
    
def save_result(I, B, D, x, y, description_of_room, 
                 outlocation=standard_out):
    
    room_of_student = {i: next(b for b in B if x[i][b]) for i in I}
    
    results = dict()
    results['NAME'] = [ i.name for i in I ]
    results['INSTITUTE'] = [ i.institute for i in I ]
    results['ROOM'] = [ room_of_student[i] for i in I ]
    results['ROOM THEMES'] = [ description_of_room[ room_of_student[i] ] 
                              for i in I ]
    
    DF = DataFrame(results)
    DF.to_csv(outlocation)

if __name__ == "__main__":
    
    I = instance.read()
    B = generate_breakoutrooms(I)
    D = distance(I)
    
    (x,y) = optimal_assignment(I, B, D)
    
    description_of_room = room_descriptions(I, B, D, x, y)
    print_result(I, B, D, x, y, description_of_room)
    save_result(I, B, D, x, y, description_of_room)
    save_match_matrix(I, D)
