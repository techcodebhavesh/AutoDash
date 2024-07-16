import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class testtt {

    public void insert(int x, int y, int player, ArrayList<ArrayList<Integer>> board) {
        if (board.get(x).get(y) == 0) {
            board.get(x).set(y, player);
        } else {
            System.out.println("Invalid move");
        }
    }

    public static int checkWin(ArrayList<ArrayList<Integer>> board) {
        // Check rows
        for (int i = 0; i < 3; i++) {
            if (board.get(i).get(0) != 0 && board.get(i).get(0).equals(board.get(i).get(1)) && board.get(i).get(1).equals(board.get(i).get(2))) {
                return board.get(i).get(0);
            }
        }
        // Check columns
        for (int i = 0; i < 3; i++) {
            if (board.get(0).get(i) != 0 && board.get(0).get(i).equals(board.get(1).get(i)) && board.get(1).get(i).equals(board.get(2).get(i))) {
                return board.get(0).get(i);
            }
        }
        // Check diagonals
        if (board.get(0).get(0) != 0 && board.get(0).get(0).equals(board.get(1).get(1)) && board.get(1).get(1).equals(board.get(2).get(2))) {
            return board.get(0).get(0);
        }
        if (board.get(0).get(2) != 0 && board.get(0).get(2).equals(board.get(1).get(1)) && board.get(1).get(1).equals(board.get(2).get(0))) {
            return board.get(0).get(2);
        }
        return 0;
    }

    public void printBoard(ArrayList<ArrayList<Integer>> board) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print(board.get(i).get(j) + " ");
            }
            System.out.println();
        }
    }

    public boolean isGameOver(ArrayList<ArrayList<Integer>> board) {
        if (checkWin(board) != 0) {
            return true;
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board.get(i).get(j) == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    public void MachineMove(ArrayList<ArrayList<Integer>> board) {

        //machine move should  be  played  accordingly thingkingly   by  maximizing
    


        int bestScore = Integer.MIN_VALUE;
        int bestMoveX = -1;
        int bestMoveY = -1;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board.get(i).get(j) == 0) {
                    board.get(i).set(j, 2);
                    int score = minimax(board, 0, false);
                    board.get(i).set(j, 0);
                    if (score > bestScore) {
                        bestScore = score;
                        bestMoveX = i;
                        bestMoveY = j;
                    }
                }
            }
        }
        board.get(bestMoveX).set(bestMoveY, 2);
    }

    public int minimax(ArrayList<ArrayList<Integer>> board, int depth, boolean isMaximizingPlayer) {
        int boardVal = evaluateBoard(board);
        if (Math.abs(boardVal) == 10 || depth == 0 || isGameOver(board)) {
            return boardVal;
        }

        if (isMaximizingPlayer) {
            int highestVal = Integer.MIN_VALUE;
            for (int i = 0; i < board.size(); i++) {
                for (int j = 0; j < board.get(i).size(); j++) {
                    if (board.get(i).get(j) == 0) { // If cell is empty
                        board.get(i).set(j, 1); // Assume 1 is the AI player
                        highestVal = Math.max(highestVal, minimax(board, depth - 1, false));
                        board.get(i).set(j, 0); // Undo move
                    }
                }
            }
            return highestVal;
        } else {
            int lowestVal = Integer.MAX_VALUE;
            for (int i = 0; i < board.size(); i++) {
                for (int j = 0; j < board.get(i).size(); j++) {
                    if (board.get(i).get(j) == 0) { // If cell is empty
                        board.get(i).set(j, 2); // Assume 2 is the opponent
                        lowestVal = Math.min(lowestVal, minimax(board, depth - 1, true));
                        board.get(i).set(j, 0); // Undo move
                    }
                }
            }
            return lowestVal;
        }
    }

    public int evaluateBoard(ArrayList<ArrayList<Integer>> board) {
        int winner = checkWin(board);
        if (winner == 1) {
            return -10;
        } else if (winner == 2) {
            return 10;
        } else {
            return 0;
        }
    }

    public static void main(String[] args) {
        ArrayList<ArrayList<Integer>> matrix = new ArrayList<>();

        // Initialize the board with all zeros
        for (int i = 0; i < 3; i++) {
            ArrayList<Integer> row = new ArrayList<>();
            for (int j = 0; j < 3; j++) {
                row.add(0);
            }
            matrix.add(row);
        }

        testtt t = new testtt();
        Scanner sc = new Scanner(System.in);
        int player = 1;
        while (!t.isGameOver(matrix)) {
            t.printBoard(matrix);
            if (player == 1) {
                System.out.println("Enter your move (row and column):");
                int x = sc.nextInt();
                int y = sc.nextInt();
                t.insert(x, y, player, matrix);
            } else {
                t.MachineMove(matrix);
            }
            player = 3 - player; // Switch player: 1 becomes 2, 2 becomes 1
        }
        t.printBoard(matrix);
        int winner = t.checkWin(matrix);
        if (winner == 1) {
            System.out.println("Player 1 wins");
        } else if (winner == 2) {
            System.out.println("Player 2 wins");
        } else {
            System.out.println("Draw");
        }
    }
}
