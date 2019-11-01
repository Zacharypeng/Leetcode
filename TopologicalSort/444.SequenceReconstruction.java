/*
    Notice the edge cases for comfirm this is the unique solution. 
    Also we have to store List instead of Set here
    because duplicate number could show more than once here
    Topological Sort:
        for the map, key should be the first, value should be the following
        think of searching for next node when stuck at this.
*/
class Solution {
    public boolean sequenceReconstruction(int[] org, List<List<Integer>> seqs) {
        Map<Integer, Integer> inDegree = new HashMap<>();
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (List<Integer> seq: seqs) {
            for (int i = 0; i < seq.size(); i++) {
                inDegree.putIfAbsent(seq.get(i), 0);
                map.putIfAbsent(seq.get(i), new ArrayList<>());
                if (i > 0) {
                    map.get(seq.get(i-1)).add(seq.get(i));
                    inDegree.put(seq.get(i), inDegree.get(seq.get(i))+1);
                }
            }            
        }
        if (org.length != inDegree.size()) return false;
        Queue<Integer> q = new LinkedList<>();
        for (int key: inDegree.keySet()) {
            if (inDegree.get(key) == 0) q.add(key);
        }
        int idx = 0;
        while (!q.isEmpty()) {
            if (q.size() > 1) return false;            
            int cur = q.poll();
            if (org[idx++] != cur) return false;
            for (int next: map.get(cur)) {
                inDegree.put(next, inDegree.get(next)-1);
                if (inDegree.get(next) == 0) q.add(next);
            }
        }
        return idx == org.length;
    }
}