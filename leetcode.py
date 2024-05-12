class Solution:
    def snakesAndLadders(self, board):

        def dfs(x, visited):
            if x==n**2 or visited.get(x, 0):
                return visited.get(x, 0)
            visited[x] = float('inf')
            for y in range(x+1, min(n**2, x+6)+1):
                row = n-1-(y-1)//n
                col = n-1-(y-1)%n if ((y-1)//n)%2 else (y-1)%n
                if not visited.get(y, 0):
                    if board[row][col]==-1:
                        visited[y] = dfs(y, visited)
                    else:
                        if board[row][col]==y:
                            visited[y]=float('inf')
                        else:
                            visited[y] = dfs(board[row][col], visited)
                visited[x] = min(visited[x], 1+visited[y])
            return visited[x]




        visited = {}
        n = len(board)
        return dfs(1, visited)

s = Solution()
board = [[-1,-1,2,-1],[14,2,12,3],[4,9,1,11],[-1,2,1,16]]
print(s.snakesAndLadders(board))