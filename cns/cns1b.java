import java.util.*;

public class cns1b {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        String str1 = "abcdefghijklmnopqrstuvwxyz";
        String str2 = "qwertyuiopasdfghjklzxcvbnm"; 

        System.out.print("Enter text: ");
        String text = sc.nextLine();

        char[] arr = text.toCharArray();

        // Encryption
        for (int i = 0; i < arr.length; i++) {
            int index = str1.indexOf(arr[i]);
            arr[i] = str2.charAt(index);
        }
        String encrypted = new String(arr);

        // Decryption
        for (int i = 0; i < arr.length; i++) {
            int index = str2.indexOf(arr[i]);
                arr[i] = str1.charAt(index);
        }

        String decrypted = new String(arr);

        System.out.println("Encrypted: " + encrypted);
        System.out.println("Decrypted: " + decrypted);

        sc.close();
    }
}