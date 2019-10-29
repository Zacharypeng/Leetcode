/*
    Dijkstra for searching the shortest path
    every element in PriorityQueue is a state -> to reach current number of workers, cost and the bikes taken
    NOTICE:
    1. use a bit map to store all the used bikes in this state
    2. use a set to store the visited state
*/
class Solution {
    class State {
        int cost, workerNum, taken;
        State(int cost, int workerNum, int taken) {
            this.cost = cost;
            this.workerNum = workerNum;
            this.taken = taken;
        }
    }
    public int assignBikes(int[][] workers, int[][] bikes) {
        Queue<State> pq = new PriorityQueue<>((a, b) -> (a.cost - b.cost));
        Set<String> set = new HashSet<>();
        pq.add(new State(0, 0, 0));
        while (!pq.isEmpty()) {
            State temp = pq.poll();
            if (temp.workerNum == workers.length) return temp.cost;
            String str = "$" + temp.workerNum + "$" + temp.taken;
            if (set.contains(str)) continue;
            set.add(str);
            for (int i = 0; i < bikes.length; i++) {
                if ( (temp.taken & (1 << i)) == 0) {
                    int dis = 
                        Math.abs(workers[temp.workerNum][0] - bikes[i][0]) + Math.abs(workers[temp.workerNum][1] - bikes[i][1]);
                    pq.add(new State(temp.cost + dis, temp.workerNum+1, temp.taken | (1 << i)));
                }
            }
        }
        return -1;
    }
}