/*
    Typical Dijkstra
    Ust a Map to represent the graph is faster
    when searching for next destination
*/
class Solution {
    public int networkDelayTime(int[][] times, int N, int K) {        
        Queue<int[]> pq = new PriorityQueue<>((a, b) -> (a[0]-b[0]));
        Map<Integer, Map<Integer, Integer>> map = new HashMap<>();
        for (int[] time: times) {
            map.putIfAbsent(time[0], new HashMap<Integer, Integer>());
            map.get(time[0]).put(time[1], time[2]);            
        }        
        boolean[] visited = new boolean[N+1];
        pq.add(new int[]{0, K});
        while (!pq.isEmpty()) {
            int[] cur = pq.poll();
            if (visited[cur[1]]) continue;            
            visited[cur[1]] = true;
            N--;
            if (N == 0) return cur[0]; 
            if (map.containsKey(cur[1])) {
                for (Map.Entry entry: map.get(cur[1]).entrySet()) {
                    int next = (int) entry.getKey();
                    int cost = (int) entry.getValue();
                    if (!visited[next])
                        pq.add(new int[]{cost + cur[0], next});
                }   
            }            
        }
        return -1;
    }
}