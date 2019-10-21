/*
    Union Find with Ranking
    when union two set, merge the smaller set into the larger set
    Notice:
    To set the parent: root[parentP] = parentQ !
    
*/


class Solution {
    
    class UnionFind {
        int[] root;
        int[] rank;
        int cnt;
        
        UnionFind(int m, int n) {
            root = new int[m*n];
            rank = new int[m*n];
            Arrays.fill(root, -1);
            Arrays.fill(rank, 1);
            cnt = 0;
        }
        
        public void union(int p, int q) {
            int parentP = find(p);
            int parentQ = find(q);
            if (parentP == parentQ) return;
            this.cnt--;
            if (rank[parentP] > rank[parentQ]) root[parentQ] = parentP;
            else if (rank[parentP] < rank[parentQ]) root[parentP] = parentQ;
            else {
                root[parentP] = parentQ;
                rank[root[p]] += rank[root[q]];
            }
            // root[parentP] = parentQ;            
        }
        
        public int find(int r) {
            if (root[r] != r) {
                root[r] = find(root[r]);                
            }            
            return root[r];
            // while (r != root[r]) {
            //     r = root[r];
            //     root[r] = root[root[r]];    
            // }
            // return r;
        }
    }
    
    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> ans = new ArrayList<>();
        UnionFind unionFind = new UnionFind(m, n);
        int[][] dirs = new int[][] {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] position: positions) {
            int pos = position[0] * n + position[1];
            if (unionFind.root[pos] != -1) {
                ans.add(unionFind.cnt);
                continue;
            }
            unionFind.root[pos] = pos;
            unionFind.cnt++;
            for (int[] dir: dirs) {
                int a  = position[0] + dir[0];
                int b = position[1] + dir[1];
                int neighbor = a * n + b;
                if (a < 0 || a >= m || b < 0 || b >= n || unionFind.root[neighbor] == -1) continue;
                unionFind.union(pos, neighbor);
            }
            ans.add(unionFind.cnt);
        }
        return ans;
    }
}