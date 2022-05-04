from collections import defaultdict
from itertools import combinations,permutations


class Solution:
    def findTarget(self, root ,k) :
        target = []

        def dfs(node):
            flag = False
            if node.val in target:
                flag = True
                return flag
            else:
                target.append(k-node.val)
                if node.left:
                    flag = dfs(node.left)
                if node.right:
                    flag = dfs(node.right)
            return flag

        flag = dfs(root)
        return flag


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

new_node2 = TreeNode(2)

new_node4 = TreeNode(4)


new_node7 = TreeNode(7)
new_node6 = TreeNode(6,None,new_node7)
new_node3 = TreeNode(3,new_node2,new_node4)
new_node5 = TreeNode(5,new_node3,new_node6)
a = Solution()
print(a.findTarget(new_node5,9))
