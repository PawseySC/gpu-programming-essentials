program cpu_vector_add
    implicit none
    integer, parameter :: n = 1000
    real(kind = 8), dimension(n) :: a, b, c

    call random_number(a)
    call random_number(b)

    b = a + b
    
end program cpu_vector_add