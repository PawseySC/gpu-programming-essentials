program reduction_cpu
    implicit none

    call test()

contains

    ! function "reduction" - TO BE COMPLETED
    ! Sums all the elements of a vector v of n elements, where n is a power of 2,
    ! using the parallel reduction algorithm.

    function reduction(v, n)
        implicit none
        integer, intent(in) :: n
        integer, intent(in), dimension(n) :: v
        integer :: reduction
        integer, dimension(n) :: tmp
        integer :: stride, thread_id

        tmp = v
        stride = ! <insert value>
        do while(stride >= !<insert value> )
            do thread_id = 1, !<insert value>
                tmp(thread_id) = tmp(thread_id) + ! complete here
            end do
            stride = ! complete here
        end do

        reduction = tmp(1)
    end function reduction


    subroutine test()
        implicit none
        integer :: v1(8), v2(16)

        v1 = [1, 1, 1, 1, 1, 1, 1, 1]

        if(reduction(v1, 8) /= 8) then
            write(*, *) "Error, reduction on v1 is not correct!"
            stop
        end if


        v2 = (/ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 /)
        if(reduction(v2, 16) /= 136) then
            write(*, *) "Error, reduction on v1 is not correct!"
            stop
        end if
    end subroutine test

end program reduction_cpu