import java.util.*;

public class cns1a {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter text: ");
        String text = sc.nextLine();

        System.out.print("Enter key (k): ");
        int k = sc.nextInt();

        char[] arr = text.toCharArray();

        // Encryption
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (char) ((arr[i] - 'a' + k) % 26 + 'a');
        }
        String encrypted = new String(arr);

        // Decryption
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (char) ((arr[i] - 'a' - k + 26) % 26 + 'a');
        }
        String decrypted = new String(arr);

        System.out.println("Encrypted: " + encrypted);
        System.out.println("Decrypted: " + decrypted);

        sc.close();
    }
}