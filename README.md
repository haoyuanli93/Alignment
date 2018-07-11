# Alignment
Align two object with similar structures

# Structure
In this repo, global_IoU_search.py and local_IoU_search.py are the two 
scripts what can be used to search for the right rotation and transformation.
The concrete usage will be updated later.

# How to use this repo
The basic strategy here is to do multiple turns of searching. The searching strategy
is tightly connected to the parametrization used here for the SO(3) group.

Here, a rotation is decided by the rotation angle and rotation angle. Therefore, 
I firstly sample through the S^2 sphere for the rotation axis. Then I sample the
[0, 2pi] for the rotation angles.   
 
The global searching is a coarse search. It will save it's optimal result. 

Then one need to use the local searching script to search around the identity 
matrix. One subtle point here is that one does not necessarily need to increase
the sampling points on S2. Also one does not necessarily increase the searching 
range for the rotation degree. The reason is that: 

The fine searching is to some degree, similar to gradient descent. One specifies 
the direction then the program will search along that direction and find the optimal
position and starting from that position and continues the search.

Therefore, one does not need a very dense sampling of the rotation group. It is just
that the search step can not decrease too slow or too quickly.
