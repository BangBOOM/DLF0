# examples/04-sendrecv.jl
using MPI
MPI.Init()


function reduce_scatter(tensor, comm, rank, size) 
end

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank + 1, size)
src = mod(rank - 1, size)

N = 4

grident = collect(1:4)

# send_mesg = Array{Float64}(undef, N)
# recv_mesg = Array{Float64}(undef, N)

recv_mesg = Array{Int64}(undef, 1)

# fill!(send_mesg, Float64(rank))
print("$rank result is $grident \n\n")
for i in 0:N-2
    rreq = MPI.Irecv!(recv_mesg, comm; source=src, tag=src + 32)
    send_idx = mod(rank - i, N) + 1
    sreq = MPI.Isend(grident[send_idx:send_idx], comm; dest=dst, tag=rank + 32)
    stats = MPI.Waitall([rreq, sreq])
    rec_idx = send_idx - 1
    if rec_idx == 0
        rec_idx = N
    end
    grident[rec_idx] += recv_mesg[1]
end
print("$rank result is $grident \n")

MPI.Barrier(comm)