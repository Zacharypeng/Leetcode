/*
    Loop from index 1 to the end, nextInt(idex + 1)
    Every element from index 0 to index get the same probability of 1 / (index + 1) to be chosen
*/

class Solution {

    int[] numCopy;
    Random random;
    
    public Solution(int[] nums) {
        numCopy = nums;
        random = new Random();
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return numCopy;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        int[] a = numCopy.clone();
        for (int i = 1; i < a.length; i++) {
            int j = random.nextInt(i+1);
            swap(a, i, j);
        }
        return a;
    }
    
    private void swap(int[] a, int i, int j) {
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int[] param_1 = obj.reset();
 * int[] param_2 = obj.shuffle();
 */