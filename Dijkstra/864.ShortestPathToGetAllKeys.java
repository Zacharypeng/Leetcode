/*
    Not Dijkstra, but the notion of using a bit map to remember a State it's similar
*/
class Solution {
    class State {
        int key;
        int row;
        int col;
        State(int k, int r, int c) {
            this.key = k;
            this.row = r;
            this.col = c;
        }
    }
    public int shortestPathAllKeys(String[] grid) {        
        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};        
        int startRow = -1, startCol = -1, max = -1;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length(); j++) {                
                char c = grid[i].charAt(j);
                if (c == '@') {
                    startRow = i;
                    startCol = j;
                }
                else if (c >= 'a' && c <= 'f') {
                    max = Math.max(c - 'a' + 1, max);
                }
            }             
        }                
        State startState = new State(0, startRow, startCol);
        Set<String> set = new HashSet<>();
        set.add(0 + " " + startRow + " " + startCol);
        Queue<State> queue = new LinkedList<>();
        queue.add(startState);
        int step = 0;        
        while (!queue.isEmpty()) {            
            int size = queue.size();                  
            while (size-- > 0) {
                State cur = queue.poll();                
                if (cur.key == (1 << max) - 1) return step;
                for (int[] dir: dirs) {
                    int row = cur.row + dir[0];
                    int col = cur.col + dir[1];
                    if (row >= 0 && row < grid.length && col >= 0 && col <grid[0].length()) {
                        char c = grid[row].charAt(col);
                        int key = cur.key; 
                        if (c >= 'a' && c <='f') {
                            key |= 1 << (c - 'a');
                        } 
                        if (c == '#') continue;
                        if (c >= 'A' && c <='F' && ((key >> (c - 'A')) & 1) == 0) continue;
                        String seen = key + " " + row + " " + col;
                        if (!set.contains(seen)) {
                            set.add(seen);
                            queue.add(new State(key, row, col));
                        }
                    }
                }
            }
            step++;
        }
        return -1;
    }
}