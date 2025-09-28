## Problem: Boss Fight
## There are N warriors, the ith of which has a health of H{i} units and can 
## deal D{i} units of damage per second. The are confronting a boss who has
## unlimited health and can deal B units of damage per second. Both the 
## warriors and the boss deal damage continuously - for example, in half a
## second, the bossdeals B/2 units of damage.

## The warriors feel it would be unfair for many of them to fight the boss at 
## once, so they'll select just two representatives to go into battle. One 
## warrior {i} will be the front line and a different warrior {j} will back
## them up. During the battle, the boss will attack warrior {i} until that
## warrior is defeated (that is until the boss has dealt H{i} units of damage to
## them), and will then attack warrior {j} until that warrior is also defeated,
## at which point the battle will end. Along the way, each of the two warriors
## will do damage to the boss as long as they are undefeated.

## Of course, the warriors will never prevail, but they'd like to determine the 
## maximum amount of damage they could deal to the boss for any choice of warriors
## {i} and {j} before the battle ends.

## Constraints:
## 2 <= N <= 500,000
## 1 <= H{i} <= 1,000,000,000
## 1 <= D{i} <= 1,000,000,000
## 1 <= B <= 1,000,000,000

## Solution
## Time Complexity: O(N*N)
## Space Complexity: O(1)
## Explanation: Despite the worst-case time complexity of O(N^2), this algorithm
## usually finds the correct solution in O(N). We can make really good guesses
## about the best warriors simply by picking a random warrior {A}, and finding
## the best warrior {B} to partner warrior {A}. Repeat this process with warrior
## {B} to find warrior {C} and so on until the maximum damage stops increasing.

# See: https://leetcode.com/discuss/post/1332986/facebook-2021-online-round-question-by-a-xc6e/comments/2158383/?parent=1317883
# Very good intuition, the following algorithm past all the testcases, but it is wrong.

from typing import List

def getMaxDamageDealt(N: int, H: List[int], D: List[int], B: int) -> float:
    ## Precompute H{i}*D{i} for each warrior {i}
    C = [h * d for h, d in zip(H, D)]
  
    max_damage = 0
    best_warrior = 0
  
    run = True
    while run:
        run = False
        next_best_warrior = 0
    
        for i in range(N):
            if i == best_warrior:
                continue
        
            damage = C[best_warrior] + C[i] + max(H[best_warrior] * D[i], H[i] * D[best_warrior])
            if damage > max_damage:
                run = True
                max_damage = damage
                next_best_warrior = i
    
        best_warrior = next_best_warrior
    
    return max_damage / B

from typing import List
from bisect import bisect_left
# Write any import statements here

def getMaxDamageDealt_v2(N: int, H: List[int], D: List[int], B: int) -> float:
    # TC: O(N*logN) solution using convex hull trick
    # SC: O(N)
    # Doesn't handle the case when multiple lines have the same slope
    
    # 1. each person is a line: y = d_j*x+d_j*h_j
    # 2. find the max convex hull formed by all lines (j=0...n-1)
    # 3. for all the line section [x_i, x_{i+1}], we have an active line
    # 4. confined to the [h_min, h_max], so that we can do binary search
    # How to solve this
    # 1. sort the lines based on slopes (D_j)
    # 2. starting from the largest one, using a stack to maintain the points, when new line start to dominate the previous line.
    # 3. if another new line get higher intersection points, pop the line out of the stack
    # 4. what are left in the stack are the active lines with a monotonic intersection points
    n, hs, ds, b = N, H, D, B

    def _intcp(d, h):
        return d*h
    
    dnis = [(d, i) for i, d in enumerate(ds)]
    dnis.sort(reverse=True)
    n_dnis = []
    for dni in dnis:
        if (not n_dnis) or n_dnis[-1][0]>dni[0]:
            n_dnis.append(dni)
        else: # n_dnis[-1][0]==dni[0]
            if _intcp(n_dnis[-1][0], hs[n_dnis[-1][1]])<_intcp(dni[0], hs[dni[1]]):
                n_dnis.pop()
                n_dnis.append(dni)
    ds = [e[0] for e in n_dnis]
    hs = [hs[e[1]] for e in n_dnis]
    m = len(ds)

    def _find_inter_x(i, j):
        return (_intcp(ds[i], hs[i])-_intcp(ds[j], hs[j]))/(ds[j]-ds[i])
    
    def _build_cvx_hull():
        # stk_lns initially stores the index of line with largest slope and intercept
        stk_lns, stk_xs = [0], []
        for i in range(1, m):
            if not stk_xs:
                x = _find_inter_x(stk_lns[-1], i)
                stk_lns.append(i) # initially missed this line
                stk_xs.append(x)
            else: # at least 2 lines in stk_lns
                while stk_xs:
                    x = _find_inter_x(stk_lns[-1], i)
                    if x>=stk_xs[-1]:
                        stk_lns.pop()
                        stk_xs.pop()
                    else: break
                x = _find_inter_x(stk_lns[-1], i) # initially missed this line
                stk_lns.append(i)
                stk_xs.append(x)
        
        assert len(stk_lns)==len(stk_xs)+1
        # print(len(stk_lns), len(stk_xs))
        # print(stk_lns, stk_xs)
        for i, x in enumerate(stk_xs[:-1]):
            if stk_xs[i]<=stk_xs[i+1]:
                raise ValueError("error")

        cvx_hull, pre_x = [], float("-inf")
        while stk_xs:
            ln = stk_lns.pop()
            x = stk_xs.pop()
            cvx_hull.append([ln, [pre_x, x]])
            pre_x = x
        assert len(stk_lns)==1 and stk_lns[0]==0
        cvx_hull.append([stk_lns[0], [pre_x, float("inf")]])
        return cvx_hull

    cvx_hull = _build_cvx_hull()
    ln_sec_exs = [ele[1][1] for ele in cvx_hull]

    # print(list(zip(ds, hs)))
    # find the max damage
    ma_d, offset = 0, 50
    for i in range(m):
        di, hi = ds[i], hs[i]
        ln_i = bisect_left(ln_sec_exs, hi)
        j = cvx_hull[ln_i][0]
        for jj in range(j-offset, j+offset+1):
            if jj==i or not 0<=jj<m: continue
            dj, hj = ds[jj], hs[jj]
            ma_d = max(ma_d, (di*hi+dj*(hi+hj))/b)
    
    return ma_d

from typing import List
from bisect import bisect_left
from collections import defaultdict

from bisect import bisect_left
from collections import defaultdict
from typing import List, Tuple

def getMaxDamageDealt_v3(N: int, H: List[int], D: List[int], B: int) -> float:
    # TC: O(N*logN) solution using convex hull trick
    # SC: O(N)
    # Handle the case when multiple lines have the same slope
    
    # - Past `24/24` testcases.
    # - After getting the runner-up correct (the second best slope), I past `22/24` testcases.
    # - After unconditionally calculating all the second best slope, I got all testcases past. Note that the second best slope can have the same `di` and `hi` as the best slope, but they are different persons.
    
    # 1. each person is a line: y = d_j*x+d_j*h_j
    # 2. find the max convex hull formed by all lines (j=0...n-1)
    # 3. for all the line section [x_i, x_{i+1}], we have an active line
    # 4. confined to the [h_min, h_max], so that we can do binary search
    # How to solve this
    # 1. sort the lines based on slopes (D_j)
    # 2. starting from the largest one, using a stack to maintain the points, when new line start to dominate the previous line.
    # 3. if another new line get higher intersection points, pop the line out of the stack
    # 4. what are left in the stack are the active lines with a monotonic intersection points
    n, o_hs, o_ds, b = N, H, D, B

    def _intcp(d, h):
        return d*h
    
    dnis = [(d, i) for i, d in enumerate(o_ds)]
    dnis.sort(reverse=True)
    n_dnis, tie_slope = [], defaultdict(list)
    for dni in dnis:
        tie_slope[dni[0]].append([dni[0], o_hs[dni[1]], dni[1]])
        if (not n_dnis) or n_dnis[-1][0]>dni[0]:
            n_dnis.append(dni)
        else: # n_dnis[-1][0]==dni[0]
            if _intcp(n_dnis[-1][0], o_hs[n_dnis[-1][1]])<_intcp(dni[0], o_hs[dni[1]]):
                n_dnis.pop()
                n_dnis.append(dni)
    # We need to consider the second best line for each slope, unconditionally                
    sec_best_slope = {}
    for di in tie_slope:
        tie_slope[di].sort(key=lambda x: x[1], reverse=True)
        if len(tie_slope[di])>1:
            sec_best_slope[di] = tie_slope[di][1]
        
    ds = [e[0] for e in n_dnis]
    hs = [o_hs[e[1]] for e in n_dnis]
    i2oi = [e[1] for e in n_dnis]
    m = len(ds)

    def _find_inter_x(i, j):
        return (_intcp(ds[i], hs[i])-_intcp(ds[j], hs[j]))/(ds[j]-ds[i])
    
    def _build_cvx_hull():
        # stk_lns initially stores the index of line with largest slope and intercept
        stk_lns, stk_xs = [0], []
        for i in range(1, m):
            if not stk_xs:
                x = _find_inter_x(stk_lns[-1], i)
                stk_lns.append(i) # initially missed this line
                stk_xs.append(x)
            else: # at least 2 lines in stk_lns
                while stk_xs:
                    x = _find_inter_x(stk_lns[-1], i)
                    if x>=stk_xs[-1]:
                        stk_lns.pop()
                        stk_xs.pop()
                    else: break
                x = _find_inter_x(stk_lns[-1], i) # initially missed this line
                stk_lns.append(i)
                stk_xs.append(x)
        
        assert len(stk_lns)==len(stk_xs)+1
        # print(len(stk_lns), len(stk_xs))
        # print(stk_lns, stk_xs)
        for i, x in enumerate(stk_xs[:-1]):
            if stk_xs[i]<=stk_xs[i+1]:
                raise ValueError("error")

        cvx_hull, pre_x = [], float("-inf")
        while stk_xs:
            ln = stk_lns.pop()
            x = stk_xs.pop()
            cvx_hull.append([ln, [pre_x, x]])
            pre_x = x
        assert len(stk_lns)==1 and stk_lns[0]==0
        cvx_hull.append([stk_lns[0], [pre_x, float("inf")]])
        return cvx_hull

    cvx_hull = _build_cvx_hull()
    ln_sec_exs = [ele[1][1] for ele in cvx_hull]

    # print(list(zip(ds, hs)))
    # find the max damage
    ma_d, offset = 0, 1 # 1 is enough
    for oi in range(n):
        di, hi = o_ds[oi], o_hs[oi]
        ln_i = bisect_left(ln_sec_exs, hi)
        j = cvx_hull[ln_i][0]
        for jj in range(j-offset, j+offset+1):
            if (not 0<=jj<m): continue
            dj, hj = ds[jj], hs[jj]
            if i2oi[jj]==oi:
                assert di==dj and hi==hj
                if dj in sec_best_slope:
                    sdj, shj, soj = sec_best_slope[dj]
                    # # The following assertion is wrong, b/c the second best slope might be have the same d and h but from a different person
                    # assert soj!=oi and sdj==di and shj!=hi
                    ma_d = max(ma_d, (di*hi+sdj*(hi+shj))/b)
            else:
                ma_d = max(ma_d, (di*hi+dj*(hi+hj))/b)
                # # The following is not needed
                # if dj in sec_best_slope:
                #     sdj, shj, soj = sec_best_slope[dj]
                #     if soj!=oi: 
                #         ma_d = max(ma_d, (di*hi+sdj*(hi+shj))/b)
    
    return ma_d

## Test Cases
if __name__ == "__main__":
    ## Test Case 1
    N = 3
    H = [2, 1, 4]
    D = [3, 1, 2]
    B = 4

    print("Test Case 1")
    print("Expected Return Value = 6.5")
    print("Actual Return Value   = {}".format(getMaxDamageDealt(N, H, D, B)))
    print("")

    ## Test Case 2
    N = 4
    H = [1, 1, 2, 100]
    D = [1, 2, 1, 3]
    B = 8
    
    print("Test Case 2")
    print("Expected Return Value = 62.75")
    print("Actual Return Value   = {}".format(getMaxDamageDealt(N, H, D, B)))
    print("")

    ## Test Case 3
    N = 4
    H = [1, 1, 2, 3]
    D = [1, 2, 1, 100]
    B = 8
    
    print("Test Case 3")
    print("Expected Return Value = 62.75")
    print("Actual Return Value   = {}".format(getMaxDamageDealt(N, H, D, B)))
    print("")
