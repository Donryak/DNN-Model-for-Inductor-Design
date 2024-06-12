#Persistent
#SingleInstance, Force
SetTitleMatchMode, 2
CoordMode, Mouse, Screen ; 마우스 좌표를 스크린 기준으로 설정

; 파라미터 범위 설정
N_Start := 1
N_End := 5
N_Step := 0.25

L1_Start := 5.0
L1_End := 10
L1_Step := 0.5

L2_Start := 6.0
L2_End := 10
L2_Step := 0.8

W_Start := 0.15
W_End := 0.2
W_Step := 0.05

S_Start := 0.2
S_End := 0.4
S_Step := 0.05

; 시작 마우스 좌표 (스크린 기준 좌표)
Start_Pos := {X: 147, Y: 302}
Sim_Start_Click_Pos := {X: 1147, Y: 136}
Sim_save_Pos := {X: 466, Y: 966} 

FilePath := "C:\Users\HOME\Desktop\DataSet\" ; 저장할 디렉토리 경로 설정

; 시뮬레이션 결과 창 이름
SimulationResultWindow := "cell_5 [Mag/Phase]:0"
LayoutWindow := "cell_5 [MyLibrary1_lib:cesll_5:layout] [EDITING] * (Layout):2"
SaveWindow := "Open CSV Export File"

; 파라미터 조합 계산 및 조건 만족 여부 확인
Loop, % (S_End - S_Start) / S_Step + 1
{
    S := S_Start + (A_Index - 1) * S_Step
    Loop, % (W_End - W_Start) / W_Step + 1
    {
        W := W_Start + (A_Index - 1) * W_Step
        Loop, % (L2_End - L2_Start) / L2_Step + 1
        {
            L2 := L2_Start + (A_Index - 1) * L2_Step
            Loop, % (L1_End - L1_Start) / L1_Step + 1
            {
                L1 := L1_Start + (A_Index - 1) * L1_Step
                Loop, % (N_End - N_Start) / N_Step + 1
                {
                    N := N_Start + (A_Index - 1) * N_Step
                    ; 조건 확인: L1과 L2는 2 × N × W + (2 × N-1) × S 보다 커야 함
                    Condition := 2 * N * W + (2 * N - 1) * S
                    if (L1 > Condition && L2 > Condition)
                    {
                        ; 파라미터 값 포맷 (소숫점 둘째자리까지)
                        N_Formatted := Format("{:.2f}", N)
                        L1_Formatted := Format("{:.2f}", L1)
                        L2_Formatted := Format("{:.2f}", L2)
                        W_Formatted := Format("{:.2f}", W)
                        S_Formatted := Format("{:.2f}", S)

                        ; 시작 좌표로 이동하여 더블 클릭
                        Click % Start_Pos.X "," Start_Pos.Y ", 2"
                        Sleep, 500 ;
                        
                        ; S 값 입력
                        Send, {Enter}
                        Send, %S_Formatted%
                        Send, {Enter 2}
                        
                        ; W 값 입력
                        Send, %W_Formatted%
                        Send, {Enter 2}
                        
                        ; L2 값 입력
                        Send, %L2_Formatted%
                        Send, {Enter 2}
                        
                        ; L1 값 입력
                        Send, %L1_Formatted%
                        Send, {Enter 2}
                        
                        ; N 값 입력
                        Send, %N_Formatted%
                        Send, {Enter}
                        
                        ; 레이아웃으로 데이터 넘기기
                        Send, !l ; Alt + L
                        Send, {Enter}
                        Sleep, 500
                        Send, {Enter}
                       
                       WinWaitActive, %LayoutWindow%,, 10
                        
                        ; 시뮬레이션 시작
                        Click % Sim_Start_Click_Pos.X "," Sim_Start_Click_Pos.Y
                        Sleep, 500 ; 안정성을 위해 잠시 대기
                        Send, {Enter}
                        
                        ; 시뮬레이션이 완료될 때까지 대기
                        WinWaitActive, %SimulationResultWindow%,, 60 ; 최대 60초 대기
                        
                        ; 결과 저장
                        Click % Sim_save_Pos.X "," Sim_save_Pos.Y ", Right"
                        Sleep, 500 ; 안정성을 위해 잠시 대기
                        Send, e ; Export 메뉴 선택
                        Send, {Enter}
                
                        
                        WinWaitActive, %SaveWindow%,, 10
                                                
                        ; 파일 경로 및 이름 설정
                        FileName := N_Formatted "_" L1_Formatted "_" L2_Formatted "_" W_Formatted "_" S_Formatted ".csv"
                        FullPath := FilePath . FileName
                        Send, %FullPath%
                        Send, {Enter}
                        
                        WinwaitClose, %SaveWindow%,, 10
                        sleep, 500
                    }
                }
            }
        }
    }
}

MsgBox, 데이터셋 생성 및 시뮬레이션 완료.
