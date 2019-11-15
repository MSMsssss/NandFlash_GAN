create trigger newblock after insert on blocks
for each row select new.block_id into @new_block_id;