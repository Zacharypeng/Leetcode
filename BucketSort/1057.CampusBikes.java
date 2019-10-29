/*
    O(MN), O(MN)
    For using bucket sort, we have to know the range of the sorting value
    here we know that the distance is in range(2001)
    We can also use greedy + priority queue( cost ASC, worker ASC, bike ASC)
*/
class Solution {
    public int[] assignBikes(int[][] workers, int[][] bikes) {
        List<int[]>[] bucket = new ArrayList[2001];
        for (int i = 0; i < workers.length; i++) {
            int[] worker = workers[i];
            for (int j = 0; j < bikes.length; j++) {
                int[] bike = bikes[j];
                int dis = Math.abs(bike[0] - worker[0]) + Math.abs(worker[1] - bike[1]);        
                if (bucket[dis] == null) bucket[dis] = new ArrayList<int[]>();
                bucket[dis].add(new int[]{i, j});
            }            
        }
        int[] ans = new int[workers.length];
        Arrays.fill(ans, -1);
        boolean[] used = new boolean[bikes.length];
        for (int i = 0; i < bucket.length; i++) {
            if (bucket[i] == null) continue;
            List<int[]> temp = bucket[i];
            for (int j = 0; j < temp.size(); j++) {
                int worker = temp.get(j)[0];
                int bike = temp.get(j)[1];
                if (!used[bike] && ans[worker] == -1) {
                    ans[worker] = bike;
                    used[bike] = true;
                }
            }
        }
        return ans;
    }
}