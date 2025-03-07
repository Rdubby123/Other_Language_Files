;******************** (C) Yifeng ZHU *******************************************
; @file    main.s
; @author  Yifeng Zhu
; @date    May-17-2015
; @note
;           This code is for the book "Embedded Systems with ARM Cortex-M 
;           Microcontrollers in Assembly Language and C, Yifeng Zhu, 
;           ISBN-13: 978-0982692639, ISBN-10: 0982692633
; @attension
;           This code is provided for education purpose. The author shall not be 
;           held liable for any direct, indirect or consequential damages, for any 
;           reason whatever. More information can be found from book website: 
;           http:;www.eece.maine.edu/~zhu/book
;*******************************************************************************

; MODE: 00: Input mode, 01: General purpose output mode
    ;       10: Alternate function mode, 11: Analog mode (reset state)

	INCLUDE core_cm4_constants.s		; Load Constant Definitions
	INCLUDE stm32l476xx_constants.s      

; Green LED <--> PA.5
LED_PIN	EQU	5

; Button <--> PC.13
BUTT_PIN EQU 13
	
	AREA    main, CODE, READONLY
	EXPORT	__main				; make __main visible to linker
	ENTRY			
				
__main	PROC
		
    ; Enable the cock to GPIO Port A, B, and C
	LDR r0, =RCC_BASE
	LDR r1, [r0, #RCC_AHB2ENR]
	ORR r1, r1, #RCC_AHB2ENR_GPIOAEN
	ORR r1, r1, #RCC_AHB2ENR_GPIOBEN
	ORR r1, r1, #RCC_AHB2ENR_GPIOCEN ; (A | B) | C
	STR r1, [r0, #RCC_AHB2ENR]
	
	; Set PA5 to output (the led)
	LDR r0, =GPIOA_BASE
	LDR r1, [r0, #GPIO_MODER]
	BIC r1, r1, #(3<<(2*LED_PIN)) ; bit clear
	ORR r1, r1, #(1<<(2*LED_PIN)) ; or
	STR r1, [r0, #GPIO_MODER] ; store modified value. Load, modify, store!
	
	; Set PA5 to push pull
	LDR r1, [r0, #GPIO_OTYPER]
	BIC r1, r1, #GPIO_OTYPER_OT_5 ; bit clear
	STR r1, [r0, #GPIO_OTYPER]
	
	; Set PA5 to no pullup no pulldown
	LDR r1, [r0, #GPIO_PUPDR]
	BIC r1, r1, #GPIO_PUPDR_PUPDR5
	STR r1, [r0, #GPIO_PUPDR]
	
	; Set PC13 to input (the button)
	LDR r0, =GPIOC_BASE
	LDR r1, [r0, #GPIO_MODER]
	BIC r1, r1, #(3<<(2*BUTT_PIN))
	STR r1, [r0, #GPIO_MODER]
	
	; Set PC13 to no pullup no pulldown
	LDR r1, [r0, #GPIO_PUPDR]
	BIC r1, r1, #GPIO_PUPDR_PUPDR13
	STR r1, [r0, #GPIO_PUPDR]

		
loop	LDR r0, =GPIOC_BASE
		LDR r1, [r0, #GPIO_IDR]
		ANDS r1, r1, #GPIO_IDR_IDR_13
		BEQ toggleLedOn
		B loop
		
toggleLEDOff	LDR r0, =GPIOC_BASE
				LDR r1, [r0, #GPIO_IDR]
				ANDS r1, r1, #GPIO_IDR_IDR_13
				BEQ toggleLEDOff
				B loop

toggleLedOn 	LDR r0, =GPIOA_BASE
				LDR r1, [r0, #GPIO_ODR]
				EOR r1, r1, #(1<<LED_PIN)
				STR r1, [r0, #GPIO_ODR]
				B toggleLEDOff
		  
		  


	ENDP
					
	ALIGN			

	AREA    myData, DATA, READWRITE
	ALIGN
array	DCD   1, 2, 3, 4
	END