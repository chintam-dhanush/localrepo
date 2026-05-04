import java.util.*;

public class cns14 {

    static int findX(int[] num, int[] rem, int k) {
        int x = 1;

        while (true) {
            int j;
            for (j = 0; j < k; j++) {
                if (x % num[j] != rem[j])
                    break;
            }

            if (j == k)
                return x;

            x++;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of equations: ");
        int k = sc.nextInt();

        int[] num = new int[k];
        int[] rem = new int[k];

        System.out.println("Enter moduli:");
        for (int i = 0; i < k; i++) {
            num[i] = sc.nextInt();
        }

        System.out.println("Enter remainders:");
        for (int i = 0; i < k; i++) {
            rem[i] = sc.nextInt();
        }

        int result = findX(num, rem, k);

        System.out.println("Smallest x is: " + result);

        sc.close();
    }
}